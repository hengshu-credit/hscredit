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
import numpy as np
import pandas as pd
from sklearn.feature_selection import chi2, SelectKBest

from .base import BaseFeatureSelector


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
    ):
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
        )
        self.k = k
        self.missing = missing
        self.method_name = "卡方检验筛选"

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

        # 处理类别变量
        X_pos = X.copy()
        for col in X_pos.columns:
            if X_pos[col].dtype == "object":
                X_pos[col] = pd.factorize(X_pos[col])[0]

        # 处理缺失值
        if self.missing is None or self.missing is False:
            mask = X_pos.notna().all(axis=1)
            X_pos = X_pos.loc[mask]
            y = np.asarray(y)[mask.values] if not isinstance(mask, np.ndarray) else np.asarray(y)[mask]
        elif isinstance(self.missing, str):
            fill_funcs = {"mean": X_pos.mean, "min": X_pos.min, "max": X_pos.max}
            if self.missing not in fill_funcs:
                raise ValueError(f"missing 仅支持 'mean'/'min'/'max'，收到: '{self.missing}'")
            X_pos = X_pos.fillna(fill_funcs[self.missing]())
        else:
            X_pos = X_pos.fillna(float(self.missing))

        # 确保非负
        X_array = np.maximum(X_pos.values, 0)

        self._validate_parallel_configuration()
        # sklearn 已能在一次矩阵调用中计算所有字段；拆成列级 joblib
        # 任务只会重复校验和调度，宽表上反而更慢。
        chi2_scores, _ = chi2(X_array, np.asarray(y))

        self.scores_ = pd.Series(chi2_scores, index=X.columns)

        # 选择特征
        if isinstance(self.k, int):
            # 保留top-k
            top_k = min(self.k, len(X.columns))
            top_indices = np.argsort(chi2_scores)[-top_k:]
            selected_cols = X.columns[top_indices].tolist()
        else:
            # 使用阈值
            selected_mask = chi2_scores >= self.threshold
            selected_cols = X.columns[selected_mask].tolist()

        self.selected_features_ = selected_cols

        # 构建详细的dropped_记录，包含卡方得分
        dropped_cols = [c for c in X.columns if c not in selected_cols]
        if len(dropped_cols) > 0:
            if isinstance(self.k, int):
                # top-k模式
                reason = f"未进入前{self.k}名"
            else:
                reason = f"卡方得分 < {self.threshold}"
            self.dropped_ = pd.DataFrame(
                {
                    "特征": dropped_cols,
                    "剔除原因": [f"{reason} (得分: {self.scores_[col]:.4f})" for col in dropped_cols],
                    "卡方得分": [self.scores_[col] for col in dropped_cols],
                }
            )
        else:
            self.dropped_ = pd.DataFrame(columns=["特征", "剔除原因", "卡方得分"])
