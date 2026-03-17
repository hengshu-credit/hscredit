"""逐步回归特征筛选器.

使用逐步回归（Stepwise Regression）进行特征筛选。
支持前向、后向、双向逐步选择，基于AIC/BIC/KS/AUC等准则进行特征取舍。

参考实现：
- toad.selection.stepwise
- sklearn.feature_selection.SequentialFeatureSelector
- statsmodels based stepwise regression
"""

from typing import Union, List, Optional, Dict, Any, Tuple
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression, LinearRegression
import warnings

from .base import BaseFeatureSelector


class StepwiseSelector(BaseFeatureSelector):
    """逐步回归特征筛选器.

    使用逐步回归方法进行特征筛选，支持前向、后向和双向选择策略。
    基于统计准则（AIC/BIC/KS/AUC）评估特征组合的优劣。

    **参数**

    :param estimator: 评估器类型，默认为'logit'
        - 'logit': 逻辑回归
        - 'ols': 最小二乘回归
        - 或传入 sklearn 兼容的评估器对象
    :param direction: 选择方向，默认为'both'
        - 'forward': 前向选择，从空模型开始逐步添加特征
        - 'backward': 后向消除，从全特征开始逐步剔除特征
        - 'both': 双向选择，结合前向和后向
    :param criterion: 筛选准则，默认为'aic'
        - 'aic': Akaike信息准则（越小越好）
        - 'bic': 贝叶斯信息准则（越小越好）
        - 'ks': Kolmogorov-Smirnov统计量（越大越好，仅适用于logit）
        - 'auc': AUC值（越大越好，仅适用于logit）
    :param max_features: 最大特征数量，默认为None（不限制）
        - 如果为整数，限制选中的特征数量
        - 如果为浮点数（0-1），限制为总特征数的比例
    :param p_enter: 前向进入阈值，默认为0.01
        - 特征加入后模型改善需要超过此阈值（对于AIC/BIC）
        - 或特征p值需要低于此阈值（对于p_value_enter）
    :param p_remove: 后向剔除阈值，默认为0.01
        - 特征剔除后模型改善需要超过此阈值
    :param p_value_enter: 双向选择中特征p值阈值，默认为0.2
        - 在双向选择中，p值超过此阈值的特征会被剔除
    :param intercept: 是否包含截距项，默认为True
    :param max_iter: 最大迭代次数，默认为100
    :param verbose: 是否打印详细信息，默认为False
    :param target: 目标变量列名，默认为'target'
    :param include: 强制保留的特征列表
    :param exclude: 强制剔除的特征列表
    :param n_jobs: 并行任务数（保留参数，暂未使用）

    **属性**

    :param selected_features_: 选中的特征列表
    :param removed_features_: 被剔除的特征列表
    :param scores_: 各特征在逐步回归中的得分
    :param history_: 逐步回归过程历史记录
    :param model_results_: 最终模型的统计结果
    :param n_features_: 选中的特征数量

    **示例**

    ::

        >>> from hscredit.core.selectors import StepwiseSelector
        >>> import pandas as pd
        >>> import numpy as np
        >>>
        >>> # 创建示例数据
        >>> np.random.seed(42)
        >>> X = pd.DataFrame({
        ...     'feature1': np.random.randn(100),
        ...     'feature2': np.random.randn(100),
        ...     'feature3': np.random.randn(100),
        ... })
        >>> y = np.random.randint(0, 2, 100)
        >>>
        >>> # 使用逐步回归筛选特征
        >>> selector = StepwiseSelector(
        ...     estimator='logit',
        ...     direction='both',
        ...     criterion='aic',
        ...     max_features=10,
        ...     p_enter=0.05,
        ...     p_remove=0.05
        ... )
        >>> selector.fit(X, y)
        >>> print(selector.selected_features_)
        ['feature1', 'feature2']
    """

    def __init__(
        self,
        estimator: Union[str, object] = 'logit',
        direction: str = 'both',
        criterion: str = 'aic',
        max_features: Optional[Union[int, float]] = None,
        p_enter: float = 0.01,
        p_remove: float = 0.01,
        p_value_enter: float = 0.2,
        intercept: bool = True,
        max_iter: int = 100,
        verbose: bool = False,
        target: str = 'target',
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        threshold: Union[float, int, str] = 0.0,
        n_jobs: int = 1,
    ):
        super().__init__(
            target=target,
            include=include,
            exclude=exclude,
            threshold=threshold,
            n_jobs=n_jobs
        )
        self.estimator = estimator
        self.direction = direction
        self.criterion = criterion
        self.max_features = max_features
        self.p_enter = p_enter
        self.p_remove = p_remove
        self.p_value_enter = p_value_enter
        self.intercept = intercept
        self.max_iter = max_iter
        self.verbose = verbose
        self.method_name = '逐步回归筛选'

    def _fit_impl(
        self,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, np.ndarray]],
    ) -> None:
        """拟合逐步回归筛选器。

        :param X: 输入特征DataFrame
        :param y: 目标变量
        """
        if y is None:
            if self.target not in X.columns:
                raise ValueError(f"需要传入y或X中包含{self.target}列")
            y = X[self.target].values
            X = X.drop(columns=self.target)

        self._get_feature_names(X)

        # 解析 max_features
        max_features = self._parse_max_features(X)

        # 移除强制剔除的特征
        X = X.drop(columns=[c for c in self.exclude_ if c in X.columns])

        # 强制保留的特征
        forced_include = [c for c in self.include_ if c in X.columns]

        # 初始化
        remaining = [c for c in X.columns if c not in forced_include]
        selected = forced_include.copy()

        # 记录历史
        self.history_ = []
        self.iteration_info_ = []

        # 根据方向初始化
        if self.direction == 'backward':
            # 后向消除：从所有特征开始
            selected = list(X.columns)
            remaining = []
            # 计算初始模型的分数
            initial_result = self._fit_model(X, y, selected)
            if initial_result['result'] is not None:
                best_score = initial_result['criterion']
            else:
                best_score = self._get_initial_score(y)
        else:
            # 前向/双向：从空模型开始
            best_score = self._get_initial_score(y)

        iter_count = 0
        improved = True

        while improved and iter_count < self.max_iter:
            iter_count += 1

            # 检查是否达到最大特征数
            if max_features is not None and len(selected) >= max_features:
                if self.verbose:
                    print(f"已达到最大特征数 {max_features}，停止迭代")
                break

            if self.direction == 'backward':
                # 后向消除
                improved, selected, remaining, best_score = self._backward_step(
                    X, y, selected, remaining, best_score, forced_include
                )

            else:
                # 前向选择或双向选择
                improved, selected, remaining, best_score = self._forward_step(
                    X, y, selected, remaining, best_score, max_features
                )

                if not improved:
                    if self.verbose:
                        print("前向选择无改善，停止迭代")
                    break

                # 双向选择：后向检验
                if self.direction == 'both' and len(selected) > len(forced_include):
                    selected, remaining = self._backward_check(
                        X, y, selected, remaining, forced_include
                    )

        # 最终选中的特征
        self.selected_features_ = selected

        # 计算特征得分（基于最终模型的p值）
        self._calculate_scores(X, y, selected)

        # 记录特征数量
        self.n_features_ = len(selected)

        self._drop_reason = '逐步回归筛选后被剔除'

    def _parse_max_features(self, X: pd.DataFrame) -> Optional[int]:
        """解析 max_features 参数。

        :param X: 输入特征DataFrame
        :return: 解析后的最大特征数量
        """
        if self.max_features is None:
            return None

        n_features = X.shape[1]

        if isinstance(self.max_features, int):
            if self.max_features < 1:
                raise ValueError("max_features 必须大于等于 1")
            return min(self.max_features, n_features)

        elif isinstance(self.max_features, float):
            if not 0 < self.max_features <= 1:
                raise ValueError("max_features 为浮点数时必须在 (0, 1] 范围内")
            return max(1, int(self.max_features * n_features))

        else:
            raise ValueError(f"max_features 类型错误: {type(self.max_features)}")

    def _get_initial_score(self, y) -> float:
        """获取初始模型分数。

        :param y: 目标变量
        :return: 初始分数
        """
        # 对于 AIC/BIC，初始化为正无穷（越小越好）
        # 对于 KS/AUC，初始化为负无穷（越大越好）
        if self.criterion in ['aic', 'bic']:
            return np.inf
        else:
            return -np.inf

    def _fit_model(self, X: pd.DataFrame, y, features: List[str]) -> Dict[str, Any]:
        """拟合模型并返回统计结果。

        :param X: 输入特征DataFrame
        :param y: 目标变量
        :param features: 要使用的特征列表
        :return: 包含模型结果的字典
        """
        if len(features) == 0:
            # 空模型（只有截距）
            X_model = np.ones((len(y), 1))
        else:
            X_model = X[features].values
            if self.intercept:
                X_model = sm.add_constant(X_model)

        try:
            # 根据评估器类型拟合模型
            if self.estimator == 'logit':
                model = sm.Logit(y, X_model)
                result = model.fit(disp=0, method='newton', maxiter=100)
            elif self.estimator == 'ols':
                model = sm.OLS(y, X_model)
                result = model.fit()
            else:
                # 使用自定义评估器
                return self._fit_custom_estimator(X, y, features)

            # 计算准则
            criterion_value = self._calculate_criterion(result, y, X_model)

            return {
                'criterion': criterion_value,
                'result': result,
                'p_values': result.pvalues,
                'params': result.params,
            }

        except Exception as e:
            if self.verbose:
                print(f"模型拟合失败: {e}")
            return {
                'criterion': self._get_initial_score(y),
                'result': None,
                'p_values': None,
                'params': None,
            }

    def _fit_custom_estimator(self, X: pd.DataFrame, y, features: List[str]) -> Dict[str, Any]:
        """使用自定义评估器拟合模型。

        :param X: 输入特征DataFrame
        :param y: 目标变量
        :param features: 要使用的特征列表
        :return: 包含模型结果的字典
        """
        if len(features) == 0:
            return {
                'criterion': self._get_initial_score(y),
                'result': None,
                'p_values': None,
                'params': None,
            }

        X_model = X[features].values
        if self.intercept:
            # 对于 sklearn 评估器，使用 fit_intercept 参数
            if hasattr(self.estimator, 'fit_intercept'):
                self.estimator.fit_intercept = self.intercept
            else:
                X_model = sm.add_constant(X_model)

        try:
            self.estimator.fit(X_model, y)

            # 计算预测值
            if hasattr(self.estimator, 'predict_proba'):
                y_pred = self.estimator.predict_proba(X_model)[:, 1]
            else:
                y_pred = self.estimator.predict(X_model)

            # 计算准则值
            criterion_value = self._calculate_criterion_from_predictions(y, y_pred, len(features))

            return {
                'criterion': criterion_value,
                'result': self.estimator,
                'p_values': None,  # sklearn 评估器通常不提供p值
                'params': getattr(self.estimator, 'coef_', None),
            }

        except Exception as e:
            if self.verbose:
                print(f"自定义评估器拟合失败: {e}")
            return {
                'criterion': self._get_initial_score(y),
                'result': None,
                'p_values': None,
                'params': None,
            }

    def _calculate_criterion(self, result, y, X_model) -> float:
        """计算准则值。

        :param result: statsmodels 结果对象
        :param y: 目标变量
        :param X_model: 模型输入矩阵
        :return: 准则值
        """
        if self.criterion == 'aic':
            return result.aic
        elif self.criterion == 'bic':
            return result.bic
        elif self.criterion == 'ks':
            y_pred = result.predict(X_model)
            return self._calculate_ks(y, y_pred)
        elif self.criterion == 'auc':
            y_pred = result.predict(X_model)
            return self._calculate_auc(y, y_pred)
        else:
            # 默认使用 AIC
            return result.aic

    def _calculate_criterion_from_predictions(self, y_true, y_pred, n_features: int) -> float:
        """从预测值计算准则值。

        :param y_true: 真实标签
        :param y_pred: 预测值
        :param n_features: 特征数量
        :return: 准则值
        """
        if self.criterion == 'ks':
            return self._calculate_ks(y_true, y_pred)
        elif self.criterion == 'auc':
            return self._calculate_auc(y_true, y_pred)
        elif self.criterion in ['aic', 'bic']:
            # 对于自定义评估器，使用简化的 AIC/BIC 计算
            n = len(y_true)
            mse = np.mean((y_true - y_pred) ** 2)
            llf = -n / 2 * np.log(2 * np.pi * mse * np.e)

            if self.criterion == 'aic':
                return 2 * n_features - 2 * llf
            else:
                return n_features * np.log(n) - 2 * llf
        else:
            return self._calculate_ks(y_true, y_pred)

    def _calculate_ks(self, y_true, y_pred) -> float:
        """计算KS统计量。

        :param y_true: 真实标签
        :param y_pred: 预测概率
        :return: KS统计量
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        # 按预测概率排序
        order = np.argsort(y_pred)
        y_true_sorted = y_true[order]

        # 计算累积分布
        cum_positive = np.cumsum(y_true_sorted)
        cum_negative = np.cumsum(1 - y_true_sorted)

        # 归一化
        total_positive = cum_positive[-1]
        total_negative = cum_negative[-1]

        if total_positive == 0 or total_negative == 0:
            return 0

        cum_positive = cum_positive / total_positive
        cum_negative = cum_negative / total_negative

        # KS统计量
        ks = np.max(np.abs(cum_positive - cum_negative))
        return ks

    def _calculate_auc(self, y_true, y_pred) -> float:
        """计算AUC值。

        :param y_true: 真实标签
        :param y_pred: 预测概率
        :return: AUC值
        """
        from sklearn.metrics import roc_auc_score
        try:
            return roc_auc_score(y_true, y_pred)
        except:
            return 0.5

    def _forward_step(
        self,
        X: pd.DataFrame,
        y,
        selected: List[str],
        remaining: List[str],
        best_score: float,
        max_features: Optional[int],
    ) -> Tuple[bool, List[str], List[str], float]:
        """执行前向选择步骤。

        :param X: 输入特征DataFrame
        :param y: 目标变量
        :param selected: 当前选中的特征
        :param remaining: 剩余可选特征
        :param best_score: 当前最佳分数
        :param max_features: 最大特征数量
        :return: (是否改善, 更新后的selected, 更新后的remaining, 更新后的best_score)
        """
        if not remaining:
            return False, selected, remaining, best_score

        # 检查是否达到最大特征数
        if max_features is not None and len(selected) >= max_features:
            return False, selected, remaining, best_score

        best_feature = None
        best_criterion = best_score
        test_results = []

        # 尝试每个候选特征
        for feature in remaining:
            test_features = selected + [feature]
            result = self._fit_model(X, y, test_features)

            if result['result'] is not None:
                criterion = result['criterion']
                test_results.append({
                    'feature': feature,
                    'criterion': criterion,
                    'p_value': result['p_values'][-1] if result['p_values'] is not None else 1.0,
                })

                # 判断是否改善
                improved = self._is_improvement(criterion, best_criterion)

                if improved:
                    best_criterion = criterion
                    best_feature = feature

        if best_feature is None:
            return False, selected, remaining, best_score

        # 判断改善是否显著
        if not self._is_significant_improvement(best_score, best_criterion):
            return False, selected, remaining, best_score

        # 添加最佳特征
        selected = selected + [best_feature]
        remaining = [f for f in remaining if f != best_feature]

        if self.verbose:
            print(f"步骤 {len(self.history_) + 1}: 添加特征 '{best_feature}', "
                  f"{self.criterion} = {best_criterion:.4f}, "
                  f"当前特征数: {len(selected)}")

        self.history_.append({
            'step': len(self.history_) + 1,
            'action': 'add',
            'feature': best_feature,
            'criterion': best_criterion,
            'selected': selected.copy(),
        })

        return True, selected, remaining, best_criterion

    def _backward_step(
        self,
        X: pd.DataFrame,
        y,
        selected: List[str],
        remaining: List[str],
        best_score: float,
        forced_include: List[str],
    ) -> Tuple[bool, List[str], List[str], float]:
        """执行后向消除步骤。

        :param X: 输入特征DataFrame
        :param y: 目标变量
        :param selected: 当前选中的特征
        :param remaining: 剩余可选特征
        :param best_score: 当前最佳分数
        :param forced_include: 强制包含的特征
        :return: (是否改善, 更新后的selected, 更新后的remaining, 更新后的best_score)
        """
        if len(selected) <= len(forced_include):
            return False, selected, remaining, best_score

        # 当前模型分数
        current_result = self._fit_model(X, y, selected)
        if current_result['result'] is None:
            return False, selected, remaining, best_score
        current_criterion = current_result['criterion']

        worst_feature = None
        worst_criterion = current_criterion

        # 尝试剔除每个特征（排除强制包含的）
        for feature in selected:
            if feature in forced_include:
                continue

            test_features = [f for f in selected if f != feature]
            if not test_features:
                continue

            result = self._fit_model(X, y, test_features)

            if result['result'] is not None:
                criterion = result['criterion']

                # 判断是否改善（剔除后准则值更优）
                if self._is_improvement(criterion, worst_criterion):
                    worst_criterion = criterion
                    worst_feature = feature

        if worst_feature is None:
            return False, selected, remaining, best_score

        # 判断改善是否显著
        if not self._is_significant_improvement(current_criterion, worst_criterion, threshold=self.p_remove):
            return False, selected, remaining, best_score

        # 移除最差特征
        selected = [f for f in selected if f != worst_feature]
        remaining = remaining + [worst_feature]

        if self.verbose:
            print(f"步骤 {len(self.history_) + 1}: 剔除特征 '{worst_feature}', "
                  f"{self.criterion} = {worst_criterion:.4f}, "
                  f"当前特征数: {len(selected)}")

        self.history_.append({
            'step': len(self.history_) + 1,
            'action': 'remove',
            'feature': worst_feature,
            'criterion': worst_criterion,
            'selected': selected.copy(),
        })

        return True, selected, remaining, worst_criterion

    def _backward_check(
        self,
        X: pd.DataFrame,
        y,
        selected: List[str],
        remaining: List[str],
        forced_include: List[str],
    ) -> Tuple[List[str], List[str]]:
        """在双向选择中进行后向检验，剔除不显著的特征。

        :param X: 输入特征DataFrame
        :param y: 目标变量
        :param selected: 当前选中的特征
        :param remaining: 剩余可选特征
        :param forced_include: 强制包含的特征
        :return: (更新后的selected, 更新后的remaining)
        """
        if len(selected) <= len(forced_include):
            return selected, remaining

        # 拟合当前模型
        result = self._fit_model(X, y, selected)

        if result['result'] is None or result['p_values'] is None:
            return selected, remaining

        # 检查p值
        p_values = result['p_values']
        # 跳过截距项（第一个是截距）
        feature_pvalues = dict(zip(['const'] + selected, p_values))

        # 找出p值超过阈值的特征（排除强制包含的）
        to_remove = []
        for feature in selected:
            if feature in forced_include:
                continue
            if feature in feature_pvalues:
                if feature_pvalues[feature] > self.p_value_enter:
                    to_remove.append((feature, feature_pvalues[feature]))

        # 按p值排序，从大到小剔除
        to_remove.sort(key=lambda x: x[1], reverse=True)

        for feature, pval in to_remove:
            selected = [f for f in selected if f != feature]
            remaining = remaining + [feature]

            if self.verbose:
                print(f"  双向选择: 剔除特征 '{feature}' (p-value = {pval:.4f})")

            self.history_.append({
                'step': len(self.history_) + 1,
                'action': 'both_remove',
                'feature': feature,
                'p_value': pval,
                'selected': selected.copy(),
            })

        return selected, remaining

    def _is_improvement(self, new_score: float, old_score: float) -> bool:
        """判断是否为改善。

        :param new_score: 新分数
        :param old_score: 旧分数
        :return: 是否为改善
        """
        if self.criterion in ['aic', 'bic']:
            # 越小越好
            return new_score < old_score
        else:
            # 越大越好（KS/AUC）
            return new_score > old_score

    def _is_significant_improvement(
        self,
        old_score: float,
        new_score: float,
        threshold: Optional[float] = None
    ) -> bool:
        """判断改善是否显著。

        :param old_score: 旧分数
        :param new_score: 新分数
        :param threshold: 阈值，默认使用 p_enter
        :return: 是否显著改善
        """
        if threshold is None:
            threshold = self.p_enter

        if self.criterion in ['aic', 'bic']:
            # 对于 AIC/BIC，改善值 = 旧值 - 新值
            improvement = old_score - new_score
            return improvement > threshold
        else:
            # 对于 KS/AUC，改善值 = 新值 - 旧值
            improvement = new_score - old_score
            return improvement > threshold

    def _calculate_scores(
        self,
        X: pd.DataFrame,
        y,
        selected: List[str],
    ) -> None:
        """计算特征得分。

        :param X: 输入特征DataFrame
        :param y: 目标变量
        :param selected: 选中的特征
        """
        if not selected:
            self.scores_ = pd.Series(dtype=float)
            return

        # 拟合最终模型
        result = self._fit_model(X, y, selected)

        if result['result'] is None or result['p_values'] is None:
            # 如果模型拟合失败，使用默认得分
            self.scores_ = pd.Series(1.0, index=selected)
            return

        # 使用p值作为得分（p值越小越好，得分越高）
        p_values = result['p_values']

        # 创建特征到p值的映射（跳过截距）
        scores = {}
        for i, feature in enumerate(selected):
            if i + 1 < len(p_values):
                # p值越小，得分越高（用1-p作为得分）
                scores[feature] = 1 - p_values[i + 1]
            else:
                scores[feature] = 0.5

        # 对于未选中的特征，得分为0
        all_features = list(X.columns)
        for feature in all_features:
            if feature not in scores:
                scores[feature] = 0.0

        self.scores_ = pd.Series(scores)

        # 保存模型结果
        self.model_results_ = result['result']

    def get_history_df(self) -> pd.DataFrame:
        """获取逐步回归历史的DataFrame格式。

        :return: 包含历史记录的DataFrame
        """
        if not hasattr(self, 'history_') or not self.history_:
            return pd.DataFrame()

        df = pd.DataFrame(self.history_)

        # 格式化
        if 'selected' in df.columns:
            df['n_features'] = df['selected'].apply(len)

        return df

    def summary(self) -> str:
        """生成筛选结果摘要。

        :return: 摘要文本
        """
        lines = []
        lines.append("=" * 70)
        lines.append("逐步回归筛选结果摘要")
        lines.append("=" * 70)
        lines.append(f"评估器类型: {self.estimator}")
        lines.append(f"选择方向: {self.direction}")
        lines.append(f"筛选准则: {self.criterion}")
        lines.append(f"最大特征数: {self.max_features}")
        lines.append(f"前向进入阈值: {self.p_enter}")
        lines.append(f"后向剔除阈值: {self.p_remove}")
        lines.append(f"迭代次数: {len(self.history_)}")
        lines.append("")
        lines.append(f"选中特征数: {len(self.selected_features_)}")
        lines.append(f"剔除特征数: {len(self.removed_features_)}")
        lines.append("")
        lines.append("选中的特征:")
        for i, feature in enumerate(self.selected_features_, 1):
            score = self.scores_.get(feature, 0)
            lines.append(f"  {i}. {feature} (得分: {score:.4f})")

        lines.append("")
        lines.append("剔除的特征:")
        for feature in self.removed_features_:
            lines.append(f"  - {feature}")

        lines.append("=" * 70)

        return "\n".join(lines)


class StepwiseFeatureSelector(StepwiseSelector):
    """逐步特征筛选器（别名）.

    为了保持与项目中其他筛选器命名风格一致，
    StepwiseSelector 也可用作 StepwiseFeatureSelector。
    
    这是 StepwiseSelector 的别名类，功能完全相同。
    """

    pass
