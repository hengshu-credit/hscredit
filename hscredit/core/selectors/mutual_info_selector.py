"""互信息筛选器.

使用互信息进行特征选择。

**参考样例**

>>> from hscredit.core.selectors import MutualInfoSelector
>>> import pandas as pd
>>> import numpy as np
>>> np.random.seed(42)
>>> X = pd.DataFrame(np.random.randn(1000, 5), columns=[f'f{i}' for i in range(5)])  # 5个特征
>>> y = pd.Series(np.random.randint(0, 2, 1000))  # 目标变量
>>> selector = MutualInfoSelector(threshold=0.1)  # 保留互信息>0.1的特征
>>> selector.fit(X, y)
>>> print(selector.selected_features_)
"""

from typing import Union, List, Optional, Dict, Any
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

from .base import BaseFeatureSelector


def _compute_mutual_info_feature(task):
    """编码并计算单个特征与目标的互信息。"""
    feature, series, y, n_neighbors, seed = task
    if series.dtype.name in ('object', 'category'):
        values = pd.factorize(series)[0].astype(float)
    else:
        values = pd.to_numeric(series, errors='coerce').astype(float).values
    if np.isnan(values).any():
        median = np.nanmedian(values)
        values = np.where(np.isnan(values), 0.0 if np.isnan(median) else median, values)
    score = mutual_info_classif(
        values.reshape(-1, 1),
        y,
        discrete_features=False,
        n_neighbors=n_neighbors,
        random_state=seed,
    )[0]
    return feature, score


class MutualInfoSelector(BaseFeatureSelector):
    """互信息筛选器.

    使用互信息（Mutual Information）评估特征与目标变量的相关性。
    互信息可以捕捉非线性关系。

    互信息值解释:
    - 0: 特征与目标完全独立
    - 值越大: 特征与目标的依赖关系越强

    **参数**

    :param threshold: 互信息阈值，默认为0.0
    :param n_neighbors: 邻居数，用于估计互信息，默认为3
    :param random_state: 随机种子
    :param target: 目标变量列名，默认为'target'
    :param n_jobs: 并行计算的任务数（注意：mutual_info_classif 不支持并行，此参数保留用于未来扩展）

    **参考样例**

    ::

        >>> from hscredit.core.selectors import MutualInfoSelector
        >>> import pandas as pd
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> X = pd.DataFrame(np.random.randn(1000, 5), columns=[f'f{i}' for i in range(5)])
        >>> y = pd.Series(np.random.randint(0, 2, 1000))
        >>> selector = MutualInfoSelector(threshold=0.1)
        >>> selector.fit(X, y)
        >>> print(selector.selected_features_)

    **注意**

    互信息可捕捉线性与非线性依赖，连续特征用 k 近邻法估计（``n_neighbors`` 越大方差越小、
    偏差略增），故结果依赖 ``random_state``。

    **引用**

    基于 sklearn ``mutual_info_classif``（Kraskov 等的 kNN 估计）：
    https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html ；
    Kraskov, A. et al. (2004). *Estimating mutual information.* Phys. Rev. E 69.
    """

    def __init__(
        self,
        threshold: float = 0.0,
        n_neighbors: int = 3,
        random_state: Optional[int] = 42,
        target: str = 'target',
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
            target=target, threshold=threshold, include=include,
            exclude=exclude, force_drop=force_drop, n_jobs=n_jobs,
            binner=binner, binning_params=binning_params,
            parallel_backend=parallel_backend, parallel_config=parallel_config,
        )
        self.n_neighbors = n_neighbors
        self.random_state = random_state
        self.method_name = '互信息筛选'

    def _fit_impl(
        self,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, np.ndarray]],
    ) -> None:
        """拟合互信息筛选器。

        :param X: 输入特征DataFrame
        :param y: 目标变量
        """
        if y is None:
            if self.target not in X.columns:
                raise ValueError(f"需要传入y或X中包含{self.target}列")
            y = X[self.target].values
            X = X.drop(columns=self.target)

        self._get_feature_names(X)

        tasks = []
        for ordinal, col in enumerate(X.columns):
            seed = None if self.random_state is None else int(self.random_state) + ordinal
            tasks.append((col, X[col], np.asarray(y), self.n_neighbors, seed))
        results = self._parallel_execute(
            _compute_mutual_info_feature,
            tasks,
            task_labels=X.columns,
        )
        mi_scores = np.array([score for _, score in results])

        self.scores_ = pd.Series(mi_scores, index=X.columns)

        # 选择互信息大于阈值的特征
        selected_mask = mi_scores >= self.threshold
        self.selected_features_ = X.columns[selected_mask].tolist()
        self._drop_reason = f'互信息值 < {self.threshold}'
