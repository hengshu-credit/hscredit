"""Boruta特征筛选器.

使用Boruta算法进行特征选择。

**参考样例**

>>> from hscredit.core.selectors import BorutaSelector
>>> from sklearn.ensemble import RandomForestClassifier
>>> import pandas as pd
>>> import numpy as np
>>> np.random.seed(42)
>>> X = pd.DataFrame(np.random.randn(200, 10), columns=[f'f{i}' for i in range(10)])  # 10个特征
>>> y = np.random.randint(0, 2, 200)  # 目标变量
>>> selector = BorutaSelector(
...     RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=42)  # Boruta需传入基模型
... )
>>> selector.fit(X, y)
>>> print(selector.selected_features_)
"""

from typing import Union, List, Optional, Dict, Any
import numpy as np
import pandas as pd
from sklearn.base import clone

try:
    from scipy.stats import binomtest as _binomtest

    def _binomial_greater_pvalue(hits: int, iterations: int) -> float:
        return float(_binomtest(hits, n=iterations, p=0.5, alternative="greater").pvalue)

except ImportError:  # scipy 1.5/1.6 兼容路径
    from scipy.stats import binom_test as _legacy_binom_test

    def _binomial_greater_pvalue(hits: int, iterations: int) -> float:
        return float(_legacy_binom_test(hits, n=iterations, p=0.5, alternative="greater"))


from .base import BaseFeatureSelector, get_feature_importances


class BorutaSelector(BaseFeatureSelector):
    """Boruta筛选器.

    Boruta是一种基于随机森林的特征选择算法。
    通过创建影子特征（shuffled版本），与原始特征进行比较，
    保留统计显著优于影子特征的特征。

    **参数**

    :param estimator: 随机森林评估器
    :param n_estimators: 树的数量，默认为100
    :param max_iter: 最大迭代次数，默认为100
    :param alpha: 多重检验校正前的显著性水平，默认为0.05
    :param random_state: 随机种子
    :param target: 目标变量列名，默认为'target'
    :param n_jobs: 并行计算的任务数

    **参考样例**

    ::

        >>> from hscredit.core.selectors import BorutaSelector
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> import pandas as pd
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> X = pd.DataFrame(np.random.randn(200, 10), columns=[f'f{i}' for i in range(10)])
        >>> y = np.random.randint(0, 2, 200)
        >>> selector = BorutaSelector(
        ...     RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=42)
        ... )
        >>> selector.fit(X, y)
        >>> print(selector.selected_features_)

    **注意**

    Boruta 是 all-relevant（保留所有相关特征）而非 minimal-optimal 的方法，倾向于多保留
    特征；计算量为 ``max_iter × n_estimators``，特征数多时较慢。

    **引用**

    Kursa, M. B., & Rudnicki, W. R. (2010). *Feature Selection with the Boruta
    Package.* Journal of Statistical Software, 36(11).
    https://doi.org/10.18637/jss.v036.i11
    """

    method_name = "Boruta筛选"

    def __init__(
        self,
        estimator=None,
        n_estimators: int = 100,
        max_iter: int = 100,
        random_state: Optional[int] = 42,
        alpha: float = 0.05,
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
            include=include,
            exclude=exclude,
            force_drop=force_drop,
            n_jobs=n_jobs,
            binner=binner,
            binning_params=binning_params,
            parallel_backend=parallel_backend,
            parallel_config=parallel_config,
        )
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.max_iter = max_iter
        self.alpha = alpha
        self.random_state = random_state

        # 默认使用随机森林

    def _fit_impl(
        self,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, np.ndarray]],
    ) -> None:
        """拟合Boruta筛选器。

        :param X: 输入特征DataFrame
        :param y: 目标变量
        """
        self._get_feature_names(X)

        if y is None:
            raise ValueError("BorutaSelector 需要目标变量 y")
        if (
            isinstance(self.max_iter, (bool, np.bool_))
            or not isinstance(self.max_iter, (int, np.integer))
            or int(self.max_iter) < 1
        ):
            raise ValueError("max_iter 必须是正整数")
        if (
            isinstance(self.n_estimators, (bool, np.bool_))
            or not isinstance(self.n_estimators, (int, np.integer))
            or int(self.n_estimators) < 1
        ):
            raise ValueError("n_estimators 必须是正整数")
        if isinstance(self.alpha, (bool, np.bool_)) or not 0 < float(self.alpha) < 1:
            raise ValueError("alpha 必须在 (0, 1) 范围内")

        n_samples, n_features = X.shape
        rng = np.random.RandomState(self.random_state)

        # 准备数据：编码类别变量并填充缺失值（基模型如随机森林不接受 object/NaN），
        # 保持与 chi2/mutual_info/f_test 等筛选器对原始信贷数据的鲁棒性一致
        X_prepared = X.copy()
        for col in X_prepared.columns:
            if X_prepared[col].dtype == "object":
                X_prepared[col] = pd.factorize(X_prepared[col])[0]
        if X_prepared.isna().any().any():
            X_prepared = X_prepared.fillna(X_prepared.median(numeric_only=True)).fillna(0)
        X_array = X_prepared.values
        feature_names = X.columns.tolist()

        if self.estimator is None:
            from sklearn.ensemble import RandomForestClassifier

            base_estimator = RandomForestClassifier(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
            )
        else:
            base_estimator = self.estimator
        base_estimator = self._clone_estimator_for_parallel(base_estimator)

        # 每一轮都让全部真实特征与本轮最大影子重要性比较。最终通过命中次数
        # 的单侧二项检验作决定，避免早期一次失手导致永久剔除。
        hits = np.zeros(n_features, dtype=int)
        importance_history = np.zeros((int(self.max_iter), n_features), dtype=float)
        shadow_thresholds = np.zeros(int(self.max_iter), dtype=float)
        history = []

        for iteration in range(int(self.max_iter)):
            # 各影子列必须独立打乱，不能用同一个行排列保留列间结构。
            X_shadow = np.column_stack([rng.permutation(X_array[:, index]) for index in range(n_features)])
            X_with_shadow = np.hstack([X_array, X_shadow])

            # 训练模型
            model = clone(base_estimator)
            model_params = model.get_params(deep=False) if hasattr(model, "get_params") else {}
            if "random_state" in model_params:
                base_random_state = (
                    self.random_state if self.random_state is not None else int(rng.randint(0, np.iinfo(np.int32).max))
                )
                model.set_params(random_state=(int(base_random_state) + iteration) % np.iinfo(np.int32).max)
            with self._estimator_parallel_context():
                model.fit(X_with_shadow, y)

            # 获取特征重要性（兼容所有模型类型）
            importances = np.asarray(get_feature_importances(model), dtype=float).reshape(-1)
            if importances.size != 2 * n_features:
                raise ValueError("Boruta 基模型返回的特征重要性长度与真实特征和影子特征总数不一致: " f"期望 {2 * n_features}，实际 {importances.size}")

            # 分离真实和影子特征重要性
            real_importances = importances[:n_features]
            shadow_importances = importances[n_features:]

            # 阈值：影子特征的最大重要性
            shadow_max = np.max(shadow_importances) if len(shadow_importances) > 0 else 0.0
            round_hits = real_importances > shadow_max
            hits += round_hits.astype(int)
            importance_history[iteration] = real_importances
            shadow_thresholds[iteration] = shadow_max

            # 记录历史
            history.append(
                {
                    "iteration": iteration,
                    "selected": int(round_hits.sum()),
                    "shadow_max": float(shadow_max),
                    "hits": hits.copy(),
                }
            )

        p_values = np.array(
            [_binomial_greater_pvalue(int(hit), int(self.max_iter)) for hit in hits],
            dtype=float,
        )
        corrected_alpha = float(self.alpha) / max(1, n_features)
        support = p_values <= corrected_alpha
        weak_support = (~support) & (hits > int(self.max_iter) / 2)

        self.hits_ = pd.Series(hits, index=feature_names, dtype=int)
        self.p_values_ = pd.Series(p_values, index=feature_names)
        self.support_ = pd.Series(support, index=feature_names)
        self.support_weak_ = pd.Series(weak_support, index=feature_names)
        self.decision_ = pd.Series(
            np.where(support, "confirmed", np.where(weak_support, "tentative", "rejected")),
            index=feature_names,
        )
        self.history_ = history
        self.importance_history_ = pd.DataFrame(importance_history, columns=feature_names)
        self.shadow_thresholds_ = pd.Series(shadow_thresholds, name="最大影子重要性")
        self.importance_scores_ = self.importance_history_.mean(axis=0)
        self.scores_ = self.hits_.astype(float) / int(self.max_iter)
        self.selected_features_ = [feature for feature in feature_names if bool(self.support_[feature])]
        self.tentative_features_ = [feature for feature in feature_names if bool(self.support_weak_[feature])]
        self._drop_reason = "Boruta 二项检验未显著优于影子特征"
