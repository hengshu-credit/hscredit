"""卡方筛选器.

使用卡方检验（Chi-Squared Test）评估特征与目标变量的独立性，
筛选出与目标显著相关的特征。适用于分类问题，需要非负特征值。
基于 sklearn.feature_selection.chi2 实现。

**参考样例**

>>> from hscredit.core.selectors import Chi2Selector
>>> import pandas as pd
>>> import numpy as np
>>> np.random.seed(42)
>>> X = pd.DataFrame(np.abs(np.random.randn(1000, 5)), columns=[f'f{i}' for i in range(5)])  # 非负特征（chi2要求）
>>> y = pd.Series(np.random.randint(0, 2, 1000))  # 目标变量
>>> selector = Chi2Selector(k=3)  # 选择得分最高的前3个特征
>>> selector.fit(X, y)
>>> print(selector.selected_features_)
"""

from typing import Union, List, Optional, Dict, Any
import warnings
import numpy as np
import pandas as pd
from sklearn.feature_selection import chi2
from sklearn.utils.multiclass import check_classification_targets
from scipy.stats import chi2_contingency

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


def _compute_chi2_feature(task):
    """计算单个非负特征的卡方得分和 p 值。"""
    feature, values, y = task
    scores, p_values = chi2(values.reshape(-1, 1), y)
    return feature, scores[0], p_values[0]


class Chi2Selector(BaseFeatureSelector):
    """卡方筛选器.

    使用卡方检验评估特征与目标变量的独立性。
    适用于分类问题和非负特征。

    卡方值解释:
    - 值越大: 特征与目标变量越相关

    **参数**

    :param threshold: 得分阈值，默认为0.0
    :param k: 保留的特征数，默认为'all'
    :param missing: 缺失值处理方式。数值则直接填充；字符串 ``'mean'``/``'min'``/``'max'`` 按列统计量填充；
        ``None`` 或 ``False`` 则删除含缺失值的行。默认为 ``-99.0``
    :param target: 目标变量列名，默认为'target'
    :param categorical_strategy: 默认 'contingency' 对 object/string/category/bool 使用 Pearson
        列联表检验（无连续性校正），不受类别编码或样本排列影响；'legacy_ordinal' 恢复旧版
        首次出现顺序编码再计算的口径，结果依赖样本顺序。类别缺失在默认策略中独立成类，
        missing=None/False 时与数值列统一删除含缺失的行，其他填充值不与真实类别合并。

    **参考样例**

    ::

        >>> from hscredit.core.selectors import Chi2Selector
        >>> import pandas as pd
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> X = pd.DataFrame(np.abs(np.random.randn(1000, 5)), columns=[f'f{i}' for i in range(5)])
        >>> y = pd.Series(np.random.randint(0, 2, 1000))
        >>> selector = Chi2Selector(k=3)
        >>> selector.fit(X, y)
        >>> print(selector.selected_features_)

    **注意**

    卡方检验要求特征非负（本类对负值通过 ``missing``/填充策略处理）；``k`` 与
    ``threshold`` 同时生效——先按得分阈值过滤，再取前 ``k`` 个。

    **引用**

    基于 sklearn ``chi2`` 评分：
    https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.chi2.html
    """

    method_name = "卡方检验筛选"

    def __init__(
        self,
        threshold: float = 0.0,
        k: Union[int, str] = "all",
        missing: Union[float, int, str, None, bool] = -99.0,
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
        categorical_strategy: str = "contingency",
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
        self.missing = missing
        self.categorical_strategy = categorical_strategy

    def __setstate__(self, state):
        legacy = "categorical_strategy" not in state
        super().__setstate__(state)
        if legacy:
            self.categorical_strategy = "legacy_ordinal"
            warnings.warn(
                "旧卡方筛选器保留类别序号检验；新训练建议显式使用 categorical_strategy='contingency'",
                UserWarning,
                stacklevel=2,
            )

    def _check_input(self, X, y=None):
        validate_real(self.threshold, "卡方阈值", allow_infinite=True)
        validate_k(self.k)
        if self.categorical_strategy not in {"contingency", "legacy_ordinal"}:
            raise ValueError("categorical_strategy 必须是 'contingency' 或 'legacy_ordinal'")
        if isinstance(self.missing, str):
            if self.missing not in {"mean", "min", "max"}:
                raise ValueError("missing 仅支持 'mean'/'min'/'max'、有限数值或 None/False")
        elif self.missing is not None and self.missing is not False:
            validate_real(self.missing, "missing")
        X, y = super()._check_input(X, y)
        if y is None or np.asarray(y).ndim != 1 or pd.isna(np.asarray(y)).any():
            raise ValueError("卡方筛选需要无缺失的一维分类目标 y")
        try:
            check_classification_targets(y)
        except ValueError as exc:
            raise ValueError("卡方筛选的目标 y 必须是离散分类标签，不能为连续数值") from exc
        return X, y

    def _fit_impl(
        self,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, np.ndarray]],
    ) -> None:
        """拟合卡方筛选器。

        :param X: 输入特征DataFrame（需要非负值）
        :param y: 目标变量
        """
        self._get_feature_names(X)

        record_counts(self, X)
        self.threshold_ = self.threshold
        self.score_name_, self.score_direction_ = "卡方统计量", "越大越好"
        categorical = [column for column in X if is_categorical(X[column])]
        processed = X.copy()
        if self.categorical_strategy == "legacy_ordinal":
            for column in categorical:
                processed[column] = pd.factorize(processed[column])[0]
        if self.missing is None or self.missing is False:
            mask = processed.notna().all(axis=1)
            processed = processed.loc[mask]
            y = np.asarray(y)[mask.to_numpy()]
        else:
            y = np.asarray(y)
        if processed.empty:
            raise ValueError("缺失值处理后没有可用于卡方检验的样本")
        self.effective_counts_ = pd.Series(len(processed), index=X.columns, dtype=np.int64)
        categorical_tests = categorical if self.categorical_strategy == "contingency" else []
        numeric = [column for column in X if column not in categorical_tests]
        scores = pd.Series(np.nan, index=X.columns, dtype=float)
        p_values = pd.Series(np.nan, index=X.columns, dtype=float)
        methods = pd.Series("非负数值卡方检验", index=X.columns, dtype=object)
        if numeric:
            numeric_frame = processed[numeric].astype(float)
            if isinstance(self.missing, str):
                numeric_frame = numeric_frame.fillna(getattr(numeric_frame, self.missing)())
            elif self.missing is not None and self.missing is not False:
                numeric_frame = numeric_frame.fillna(float(self.missing))
            if numeric_frame.isna().any().any():
                raise ValueError("数值字段全缺失，无法按列统计量填充；请提供有限数值 missing")
            numeric_scores, numeric_p = chi2(np.maximum(numeric_frame.to_numpy(), 0), y)
            scores.loc[numeric], p_values.loc[numeric] = numeric_scores, numeric_p
            if self.categorical_strategy == "legacy_ordinal":
                methods.loc[categorical] = "旧版类别序号卡方检验"
        target_codes, target_labels = pd.factorize(y, sort=False)
        for column in categorical_tests:
            codes, labels = pd.factorize(processed[column], sort=False, use_na_sentinel=False)
            table = np.bincount(
                codes * len(target_labels) + target_codes,
                minlength=len(labels) * len(target_labels),
            ).reshape(len(labels), len(target_labels))
            if min(table.shape) < 2:
                score, probability = 0.0, 1.0
            else:
                score, probability, _, _ = chi2_contingency(table, correction=False)
            scores[column], p_values[column] = score, probability
            methods[column] = "类别列联表卡方检验"
        self._validate_parallel_configuration()
        self.scores_, self.p_values_, self.test_methods_ = scores, p_values, methods
        self.ranks_ = rank_scores(scores)
        threshold_mask = scores.to_numpy() >= self.threshold
        quantity_mask = top_k_mask(scores.to_numpy(), self.k)
        record_conditions(self, X.columns, 阈值达标=threshold_mask, 数量限制达标=quantity_mask)
        selected_mask = threshold_mask & quantity_mask
        selected_cols = X.columns[selected_mask].tolist()

        self.selected_features_ = selected_cols

        # 构建详细的dropped_记录，包含卡方得分
        dropped_cols = [c for c in X.columns if c not in selected_cols]
        if len(dropped_cols) > 0:
            reasons = []
            for column in dropped_cols:
                parts = []
                if not np.isfinite(scores[column]):
                    parts.append("卡方统计量无效（全零或无有效变异）")
                elif not bool(self.condition_results_.loc[column, "阈值达标"]):
                    parts.append(f"卡方统计量 < {self.threshold}")
                if not bool(self.condition_results_.loc[column, "数量限制达标"]):
                    parts.append(f"未进入前{self.k}名")
                reasons.append("；".join(parts))
            self.dropped_ = pd.DataFrame(
                {
                    "特征": dropped_cols,
                    "剔除原因": reasons,
                    "卡方得分": [self.scores_[col] for col in dropped_cols],
                }
            )
        else:
            self.dropped_ = pd.DataFrame(columns=["特征", "剔除原因", "卡方得分"])
