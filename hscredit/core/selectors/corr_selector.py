"""相关性筛选器.

移除与其它特征高度相关的特征，保留指标值更优的特征。
参考 scorecardpipeline / toad 的相关性筛选逻辑。

**参考样例**

>>> from hscredit.core.selectors import CorrSelector
>>> import pandas as pd
>>> import numpy as np
>>> np.random.seed(42)
>>> X = pd.DataFrame(np.random.randn(1000, 5), columns=[f'f{i}' for i in range(5)])  # 5个特征
>>> y = pd.Series(np.random.randint(0, 2, 1000))  # 目标变量（用于IV计算）
>>> selector = CorrSelector(threshold=0.7, metric='iv')  # 相关性>0.7时移除，保留IV更高的特征
>>> selector.fit(X, y)
>>> print(selector.selected_features_)
"""

from typing import Union, List, Optional, Dict, Any
import numpy as np
import pandas as pd

from .base import BaseFeatureSelector
from ...exceptions import ValidationError


# bin_tables_ 中指标列名 → 聚合方式的映射
_METRIC_COL_MAP = {
    'iv': ('指标IV值', lambda s: s.iloc[0]),   # 每行相同，取第一个
    'ks': ('分档KS值', 'max'),                  # 取最大KS
    'lift': ('LIFT值', 'max'),                  # 取最大LIFT
    'bad_rate': ('坏样本率', 'max'),             # 取最大坏样本率
}

DEFAULT_BINNING_PARAMS = {
    'method': 'best_iv',
    'max_n_bins': 5,
    'min_bin_size': 0.01,
    'missing_separate': True,
}


class CorrSelector(BaseFeatureSelector):
    """相关性筛选器.

    移除与其它特征相关性高于阈值的特征。
    当两个特征高度相关时，保留指定指标（默认 IV）更高的特征，
    与 scorecardpipeline / toad 的相关性筛选逻辑一致。

    **参数**

    :param threshold: 相关系数阈值，默认为0.7
        - 0.7: 移除与其它特征相关性超过0.7的特征
        - 范围: 0-1之间的浮点数
    :param method: 相关性计算方法，默认为'pearson'
        - 'pearson': 皮尔逊相关系数
        - 'spearman': 斯皮尔曼等级相关系数
        - 'kendall': 肯德尔相关系数
    :param metric: 用于决定保留哪个特征的指标，默认为'iv'
        - 'iv': 信息值（需要目标变量 y）
        - 'ks': KS 统计量
        - 'lift': LIFT 值
        - 'bad_rate': 坏样本率
        指标通过分箱后的 bin_tables_ 计算得到。
    :param weights: 特征权重，用于决定保留哪个特征，默认为None
        如果同时传入 weights 和 metric，weights 优先。
    :param binning_params: 透传给 OptimalBinning 的筛选前分箱参数，例如:
        - method: 分箱方法，默认'best_iv'
        - max_n_bins: 最大分箱数，默认5
        - min_bin_size: 最小分箱比例，默认0.01
        - missing_separate: 是否缺失单独分箱，默认True
        - prebinning: 预分箱方法
        等等，详见 OptimalBinning 文档。

    **参考样例**

    >>> from hscredit.core.selectors import CorrSelector
    >>> selector = CorrSelector(threshold=0.7, metric='iv')
    >>> selector.fit(X, y)
    >>> print(selector.selected_features_)
    >>>
    >>> # 基于 KS 的相关性筛选
    >>> selector = CorrSelector(threshold=0.7, metric='ks')
    >>> selector.fit(X, y)
    >>>
    >>> # 自定义分箱参数
    >>> selector = CorrSelector(
    ...     threshold=0.7,
    ...     metric='iv',
    ...     binning_params={'method': 'cart', 'max_n_bins': 8}
    ... )
    >>> selector.fit(X, y)
    >>>
    >>> # 使用自定义权重（不做分箱）
    >>> selector = CorrSelector(threshold=0.7, weights=iv_series)
    >>> selector.fit(X)

    **引用**

    相关系数计算基于 ``pandas.DataFrame.corr``（Pearson/Spearman/Kendall）：
    https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.corr.html ；
    高相关特征对中保留 IV/KS 更高者的策略对齐 toad / scorecardpipeline 的相关性筛选。
    """

    def __init__(
        self,
        threshold: float = 0.7,
        method: str = 'pearson',
        metric: str = 'iv',
        weights: Optional[Union[pd.Series, Dict[str, float], List[float]]] = None,
        binning_params: Optional[Dict[str, Any]] = DEFAULT_BINNING_PARAMS,
        target: str = 'target',
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        force_drop: Optional[List[str]] = None,
        n_jobs: int = 1,
        binner: Optional[Any] = None,
    ):
        self._uses_default_binning_params = (
            binning_params is DEFAULT_BINNING_PARAMS
            or binning_params == DEFAULT_BINNING_PARAMS
        )
        effective_binning_params = (
            dict(DEFAULT_BINNING_PARAMS)
            if self._uses_default_binning_params
            else binning_params
        )
        super().__init__(
            target=target, threshold=threshold, include=include,
            exclude=exclude, force_drop=force_drop, n_jobs=n_jobs,
            binner=binner, binning_params=effective_binning_params,
        )
        self.method = method
        self.metric = metric
        self.weights = weights
        self.method_name = '相关性筛选'

    def _metric_weights_from_binner(self, feature_names: List[str]) -> pd.Series:
        """从基类管理的同一分箱器中提取特征指标权重。"""
        metric_key = self.metric.lower()
        if metric_key not in _METRIC_COL_MAP:
            raise ValidationError(
                f"不支持的指标 '{self.metric}'，可选: {list(_METRIC_COL_MAP.keys())}"
            )
        col_name, agg_func = _METRIC_COL_MAP[metric_key]
        binner = getattr(self, '_binner_instance', None)
        bin_tables = getattr(binner, 'bin_tables_', {}) if binner is not None else {}
        if not bin_tables:
            raise ValidationError(
                "CorrSelector 使用指标权重时需要配置 binner 或 binning_params，"
                "也可以显式传入 weights"
            )

        scores = {}
        for col in feature_names:
            if col in bin_tables:
                bt = bin_tables[col]
                if col_name in bt.columns:
                    scores[col] = bt[col_name].agg(agg_func)
                else:
                    scores[col] = 0.0
            else:
                scores[col] = 0.0

        return pd.Series(scores)

    def _should_apply_binner(
        self,
        y: Optional[Union[pd.Series, np.ndarray]],
    ) -> bool:
        """无目标时仅跳过构造函数提供的默认监督分箱。"""
        if y is None and self.binner is None and self._uses_default_binning_params:
            return False
        return super()._should_apply_binner(y)

    def _fit_impl(
        self,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, np.ndarray]],
    ) -> None:
        """拟合相关性筛选器。

        :param X: 输入特征DataFrame
        :param y: 目标变量
        """
        self._get_feature_names(X)

        n_features = X.shape[1]
        feature_names = X.columns.tolist()

        # ── 构建权重 ──
        if self.weights is not None:
            # 用户显式传入权重
            if isinstance(self.weights, pd.Series):
                weight_series = self.weights.reindex(feature_names).fillna(0.0)
            elif isinstance(self.weights, dict):
                weight_series = pd.Series(self.weights).reindex(feature_names).fillna(0.0)
            else:
                weight_series = pd.Series(
                    np.array(self.weights)[:n_features],
                    index=feature_names[:len(self.weights)],
                ).reindex(feature_names).fillna(0.0)
        elif y is not None or getattr(self, '_binner_instance', None) is not None:
            # 从基类已训练的同一分箱器中读取指标权重
            weight_series = self._metric_weights_from_binner(feature_names)
            weight_series = weight_series.reindex(feature_names).fillna(0.0)
        else:
            # 无 y 且无 weights，使用等权（退化为按列顺序保留）
            weight_series = pd.Series(np.ones(n_features), index=feature_names)

        weight_arr = weight_series.values
        self.feature_scores_ = weight_series.copy()
        self.scores_ = weight_series.copy()

        # ── 按权重降序排列（权重高的优先保留） ──
        sort_idx = np.argsort(weight_arr)[::-1]
        sorted_names = [feature_names[i] for i in sort_idx]
        sorted_weights = weight_arr[sort_idx]

        # ── 计算相关矩阵 ──
        X_sorted = X[sorted_names]
        corr_matrix = X_sorted.corr(method=self.method).abs()

        # ── 贪心剔除：扫描上三角，剔除权重低的 ──
        drops = set()
        upper = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        high_corr = np.where((corr_matrix.values > self.threshold) & upper)

        for i, j in zip(high_corr[0], high_corr[1]):
            # i 在排序后的位置比 j 小 → i 权重 >= j 权重
            # 剔除权重低的那个；如果权重相同，剔除后出现的（j）
            if sorted_weights[i] >= sorted_weights[j]:
                drops.add(j)
            else:
                drops.add(i)

        # ── 获取保留的特征 ──
        keep_idx = [idx for idx in range(n_features) if idx not in drops]
        self.selected_features_ = [sorted_names[idx] for idx in keep_idx]

        # 保存 scores（与原始列顺序一致）
        self.scores_ = weight_series

        # ── 构建 dropped_ 报告 ──
        if len(drops) > 0:
            dropped_cols = [sorted_names[idx] for idx in drops]
            max_corr_values = []
            max_corr_features = []
            metric_values = []
            for idx in drops:
                col_name = sorted_names[idx]
                corr_values = corr_matrix.loc[col_name, :].copy()
                corr_values[col_name] = 0
                max_corr = corr_values.max()
                max_corr_feat = corr_values.idxmax()
                max_corr_values.append(max_corr)
                max_corr_features.append(max_corr_feat)
                metric_values.append(weight_series.get(col_name, 0.0))

            metric_label = (
                self.metric.upper()
                if self.weights is None and getattr(self, '_binner_instance', None) is not None
                else '权重'
            )
            self.dropped_ = pd.DataFrame({
                '特征': dropped_cols,
                '剔除原因': [
                    f'与{max_corr_features[k]}相关系数({max_corr_values[k]:.4f})>={self.threshold}，'
                    f'{metric_label}({metric_values[k]:.4f})较低'
                    for k in range(len(dropped_cols))
                ],
                '最大相关系数': max_corr_values,
                '相关特征': max_corr_features,
                metric_label: metric_values,
                '阈值': [self.threshold] * len(dropped_cols),
            })
        else:
            self.dropped_ = pd.DataFrame(
                columns=['特征', '剔除原因', '最大相关系数', '相关特征', '阈值']
            )
