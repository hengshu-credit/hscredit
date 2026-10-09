"""F检验筛选器.

使用单因素方差分析（ANOVA F-Test）评估特征与目标变量的线性相关性，
筛选出组间差异显著的特征。适用于分类问题中的特征筛选。
基于 sklearn.feature_selection.f_classif 实现。

**参考样例**

>>> from hscredit.core.selectors import FTestSelector
>>> import pandas as pd
>>> import numpy as np
>>> np.random.seed(42)
>>> X = pd.DataFrame(np.random.randn(1000, 5), columns=[f'f{i}' for i in range(5)])  # 5个特征
>>> y = pd.Series(np.random.randint(0, 2, 1000))  # 目标变量
>>> selector = FTestSelector(k=3)  # 选择F检验得分最高的前3个特征
>>> selector.fit(X, y)
>>> print(selector.selected_features_)
"""

from typing import Union, List, Optional, Dict, Any
import warnings
import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif

from .base import BaseFeatureSelector
from ._statistical_utils import (
    is_categorical,
    rank_scores,
    record_conditions,
    record_counts,
    top_k_mask,
    validate_k,
    validate_real,
)


def _compute_f_test_feature(task):
    """计算单个特征的 ANOVA F 得分和 p 值。"""
    feature, values, y = task
    scores, p_values = f_classif(values.reshape(-1, 1), y)
    return feature, scores[0], p_values[0]


class FTestSelector(BaseFeatureSelector):
    """F检验筛选器.

    使用F检验（ANOVA）评估特征与目标变量的相关性。
    适用于分类问题。

    F值解释:
    - 值越大: 特征与目标变量越相关

    **参数**

    :param threshold: F值阈值，默认为0.0
    :param k: 保留的特征数，默认为'all'
    :param percentile: 保留的特征百分比，默认为None
    :param target: 目标变量列名，默认为'target'
    :param categorical_strategy: 默认 'error' 拒绝无序类别，需先做有业务含义的数值编码。
        'legacy_ordinal' 显式恢复旧版首次出现序号编码（结果可能依赖样本排列）；
        有序 category 使用声明的类别顺序，不受出现顺序影响。

    **参考样例**

    ::

        >>> from hscredit.core.selectors import FTestSelector
        >>> import pandas as pd
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> X = pd.DataFrame(np.random.randn(1000, 5), columns=[f'f{i}' for i in range(5)])
        >>> y = pd.Series(np.random.randint(0, 2, 1000))
        >>> selector = FTestSelector(k=3)
        >>> selector.fit(X, y)
        >>> print(selector.selected_features_)

    **注意**

    F 检验只能捕捉特征与目标的**线性**相关，非线性关系可能漏检（此时改用
    :class:`MutualInfoSelector`）。``k``/``percentile``/``threshold`` 可组合限制选中数量。

    **引用**

    基于 sklearn ``f_classif``（ANOVA F 值）：
    https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_classif.html
    """

    method_name = "F检验筛选"

    def __init__(
        self,
        threshold: float = 0.0,
        k: Union[int, str] = "all",
        percentile: Optional[int] = None,
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
        categorical_strategy: str = "error",
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
        self.k = k
        self.percentile = percentile
        self.categorical_strategy = categorical_strategy

    def __setstate__(self, state):
        legacy = "categorical_strategy" not in state
        super().__setstate__(state)
        if legacy:
            self.categorical_strategy = "legacy_ordinal"
            warnings.warn(
                "旧F检验筛选器保留类别序号编码；新训练建议先明确类别编码再使用 categorical_strategy='error'",
                UserWarning,
                stacklevel=2,
            )

    def _check_input(self, X, y=None):
        validate_real(self.threshold, "F检验阈值", allow_infinite=True)
        validate_k(self.k)
        if self.percentile is not None and (
            isinstance(self.percentile, (bool, np.bool_))
            or not isinstance(self.percentile, (int, np.integer))
            or not 0 < int(self.percentile) <= 100
        ):
            raise ValueError("percentile 必须是 (0, 100] 范围内的整数")
        if self.categorical_strategy not in {"error", "legacy_ordinal"}:
            raise ValueError("categorical_strategy 必须是 'error' 或 'legacy_ordinal'")
        return super()._check_input(X, y)

    def _fit_impl(
        self,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, np.ndarray]],
    ) -> None:
        """拟合F检验筛选器。

        :param X: 输入特征DataFrame
        :param y: 目标变量
        """
        if y is None:
            if self.target not in X.columns:
                raise ValueError(f"需要传入y或X中包含{self.target}列")
            y = X[self.target].values
            X = X.drop(columns=self.target)

        self._get_feature_names(X)
        record_counts(self, X)
        self.threshold_ = self.threshold
        self.score_name_, self.score_direction_ = "F统计量", "越大越好"

        if isinstance(self.k, (int, np.integer)) and not isinstance(self.k, (bool, np.bool_)):
            if int(self.k) <= 0:
                raise ValueError("k 必须大于 0")
        elif self.k != "all":
            raise ValueError("k 必须是大于 0 的整数或 'all'")
        if self.percentile is not None and (
            isinstance(self.percentile, (bool, np.bool_))
            or not isinstance(self.percentile, (int, np.integer))
            or not 0 < int(self.percentile) <= 100
        ):
            raise ValueError("percentile 必须是 (0, 100] 范围内的整数")

        # 处理类别变量
        X_encoded = X.copy()
        self.test_methods_ = pd.Series("数值ANOVA F检验", index=X.columns, dtype=object)
        for col in X.columns:
            if isinstance(X[col].dtype, pd.CategoricalDtype) and X[col].dtype.ordered:
                X_encoded[col] = X[col].cat.codes.astype(float).where(X[col].notna(), np.nan)
                self.test_methods_[col] = "有序类别ANOVA F检验"
            elif is_categorical(X[col]) and not pd.api.types.is_bool_dtype(X[col].dtype):
                if self.categorical_strategy != "legacy_ordinal":
                    raise ValueError(
                        f"F检验不支持无序类别字段 '{col}'；请先数值编码，或显式指定 categorical_strategy='legacy_ordinal'"
                    )
                X_encoded[col] = pd.factorize(X[col])[0]
                self.test_methods_[col] = "旧版类别序号ANOVA F检验"
            else:
                X_encoded[col] = X_encoded[col].astype(float)

        # 缺失值处理：f_classif 不接受 NaN，使用列中位数填充，整列缺失时回退为 0，
        # 保持与 chi2/mutual_info 等筛选器对原始信贷数据的鲁棒性一致
        if X_encoded.isna().any().any():
            X_encoded = X_encoded.fillna(X_encoded.median(numeric_only=True)).fillna(0)

        self._validate_parallel_configuration()
        # f_classif 原生支持二维矩阵，整表计算可复用中心化/校验开销。
        f_scores, p_values = f_classif(X_encoded.values, np.asarray(y))

        # 处理NaN
        self.raw_scores_ = pd.Series(f_scores, index=X.columns)
        # 完全分离对应的正无穷 F 值有明确含义，不能伪装成最大有限浮点数。
        f_scores = np.where(np.isnan(f_scores), 0.0, f_scores)

        self.scores_ = pd.Series(f_scores, index=X.columns)
        self.p_values_ = pd.Series(p_values, index=X.columns)
        self.ranks_ = rank_scores(self.scores_)
        self.effective_counts_ = self.total_counts_.copy()

        # 选择特征
        threshold_mask = f_scores >= self.threshold
        percentile_mask = np.ones(len(X.columns), dtype=bool)
        self.percentile_values_ = None
        self.percentile_cutoff_ = None
        self.percentile_tie_slots_ = None
        if self.percentile is not None:
            # 复用原始分数，匹配 SelectPercentile 的 NaN 末位及同分输入顺序规则。
            if self.percentile != 100:
                raw = self.raw_scores_.to_numpy().copy()
                raw[np.isnan(raw)] = -np.inf
                # 对严格单调的稠密序号做分位数，保留同值边界规则，同时避免
                # 完全分离的 +inf 在插值时出现 inf - inf 导致所有特征被剔除。
                order_values = pd.Series(raw).rank(method="dense").to_numpy()
                cutoff = np.percentile(order_values, 100 - self.percentile)
                percentile_mask = order_values > cutoff
                ties = np.flatnonzero(order_values == cutoff)
                remaining = int(len(raw) * self.percentile / 100) - int(percentile_mask.sum())
                self.percentile_values_ = pd.Series(order_values, index=X.columns)
                self.percentile_cutoff_ = float(cutoff)
                self.percentile_tie_slots_ = max(0, remaining)
                if remaining > 0:
                    percentile_mask[ties[:remaining]] = True
        quantity_mask = top_k_mask(f_scores, self.k)
        record_conditions(
            self, X.columns, 阈值达标=threshold_mask, 百分位达标=percentile_mask, 数量限制达标=quantity_mask
        )
        selected_mask = threshold_mask & percentile_mask & quantity_mask

        self.selected_features_ = X.columns[selected_mask].tolist()
        dropped_columns = X.columns[~selected_mask].tolist()
        reasons = []
        for column in dropped_columns:
            conditions = self.condition_results_.loc[column]
            parts = []
            if not conditions["阈值达标"]:
                parts.append(f"F值 < {self.threshold}")
            if not conditions["百分位达标"]:
                parts.append(f"未进入前{self.percentile}%")
            if not conditions["数量限制达标"]:
                parts.append(f"未进入前{self.k}名")
            reasons.append("；".join(parts))
        self.dropped_ = pd.DataFrame(
            {"特征": dropped_columns, "剔除原因": reasons, "F值": self.scores_.reindex(dropped_columns).to_numpy()}
        )
