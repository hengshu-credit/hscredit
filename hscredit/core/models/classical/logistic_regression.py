"""逻辑回归模型模块.

提供扩展的逻辑回归模型，继承自 sklearn.linear_model.LogisticRegression，
增加了统计信息计算功能，包括标准误差、z统计量、p值、置信区间和VIF等。

核心功能:
- 继承 sklearn LogisticRegression 的所有功能
- 自动计算统计信息（标准误差、z值、p值）
- 支持 VIF（方差膨胀因子）计算
- 提供 summary() 方法输出回归结果表

参考实现:
- scorecardpipeline.model.ITLubberLogisticRegression
- skorecard.linear_model.LogisticRegression

示例:
    >>> from hscredit.core.models import LogisticRegression
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
    >>> model = LogisticRegression(calculate_stats=True, C=1.0, max_iter=1000)
    >>> model.fit(X, y)
    >>> summary = model.summary()
    >>> print(summary)
"""

import numpy as np
import pandas as pd
import scipy.special
import scipy.stats
import inspect
from typing import Union, Optional, List
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.utils.validation import check_is_fitted


_SKLEARN_LOGISTIC_PARAMS = set(inspect.signature(SklearnLogisticRegression.__init__).parameters)


class LogisticRegression(SklearnLogisticRegression):
    """扩展逻辑回归模型.

    继承 sklearn.linear_model.LogisticRegression，增加统计信息计算功能。
    在训练时自动计算以下统计信息（当 calculate_stats=True 时）：
    - cov_matrix_: 参数的协方差矩阵
    - std_err_intercept_: 截距的标准误差
    - std_err_coef_: 系数的标准误差
    - z_intercept_: 截距的z统计量
    - z_coef_: 系数的z统计量
    - p_val_intercept_: 截距的p值
    - p_val_coef_: 系数的p值
    - vif_: 方差膨胀因子（VIF）

    可通过 summary() 方法获取格式化的回归结果表。

    **参数**

    :param penalty: 正则化类型，可选 'l1', 'l2', 'elasticnet', 'none'，默认 'l2'
        - 'newton-cg', 'lbfgs', 'sag': 仅支持 'l2'
        - 'liblinear', 'saga': 支持 'l1', 'l2'
        - 'saga': 支持 'elasticnet'
    :param calculate_stats: 是否在训练时计算统计信息，默认 True
        设置为 False 可提高训练速度，但无法使用 summary() 方法
    :param dual: 是否使用对偶形式，默认 False
        仅当 solver='liblinear' 时有效
    :param tol: 优化算法的收敛容差，默认 1e-4
    :param C: 正则化强度的倒数，默认 1.0
        必须是正浮点数，值越小正则化越强
    :param fit_intercept: 是否拟合截距项，默认 True
    :param intercept_scaling: 截距缩放系数，默认 1.0
        仅当 solver='liblinear' 且 fit_intercept=True 时有效
    :param class_weight: 类别权重，默认 None
        可选 'balanced' 或自定义字典
        'balanced' 使用 n_samples / (n_classes * np.bincount(y))
    :param random_state: 随机数种子，默认 None
        仅在 solver 为 'sag', 'saga', 'liblinear' 时有效
    :param solver: 优化算法，默认 'lbfgs'
        - 'newton-cg': 牛顿共轭梯度法
        - 'lbfgs': 拟牛顿法
        - 'liblinear': 坐标下降法
        - 'sag': 随机平均梯度下降
        - 'saga': SAGA 随机优化算法
    :param max_iter: 最大迭代次数，默认 100
    :param multi_class: 多分类策略，默认 'auto'
        可选 'auto', 'ovr', 'multinomial'
    :param verbose: 日志详细程度，默认 0
    :param warm_start: 是否使用上次结果初始化，默认 False
    :param n_jobs: 并行计算的CPU核心数，默认 None
        -1 表示使用所有可用核心
    :param l1_ratio: 弹性网络混合参数，默认 None
        0 <= l1_ratio <= 1，仅当 penalty='elasticnet' 时有效
    :param positive_woe_coef: WOE 模型系数正向化策略，默认 'auto'
        - 'auto': 仅当输入 DataFrame 被标记为 hscredit 的 WOE 编码结果时启用
        - True: 始终启用。会将负系数对应列乘以 -1，并把系数改为正值
        - False: 禁用，保持 sklearn 原始系数符号

    **属性**

    - coef_: 模型系数，形状 (n_classes, n_features) 或 (n_features,)
    - intercept_: 截距项，形状 (n_classes,) 或 (1,)
    - classes_: 类别标签数组
    - n_features_in_: 训练时的特征数量
    - feature_names_in_: 训练时的特征名称（当输入为DataFrame时）
    - cov_matrix_: 参数的协方差矩阵
    - std_err_coef_: 系数的标准误差
    - std_err_intercept_: 截距的标准误差
    - z_coef_: 系数的z统计量
    - z_intercept_: 截距的z统计量
    - p_val_coef_: 系数的p值
    - p_val_intercept_: 截距的p值
    - vif_: 方差膨胀因子数组
    - woe_coef_signs_: WOE 列方向调整向量，1 表示不变，-1 表示该列已翻转
    - raw_coef_: 原始拟合系数（正向化前，仅在启用 positive_woe_coef 后提供）

    **参考样例**

    基本使用::

        >>> from hscredit.core.models import LogisticRegression
        >>> import pandas as pd
        >>> import numpy as np
        >>>
        >>> # 创建示例数据
        >>> np.random.seed(42)
        >>> X = pd.DataFrame({
        ...     'age': np.random.randint(18, 65, 1000),
        ...     'income': np.random.randint(3000, 50000, 1000),
        ... })
        >>> y = (X['age'] + X['income'] / 1000 > 50).astype(int)
        >>>
        >>> # 训练模型
        >>> model = LogisticRegression(calculate_stats=True, max_iter=1000)
        >>> model.fit(X, y)
        >>>
        >>> # 查看统计摘要
        >>> summary = model.summary()
        >>> print(summary[['Coef.', 'Std.Err', 'P>|z|', 'VIF']])

    使用样本权重::

        >>> sample_weight = np.where(y == 1, 2.0, 1.0)  # 增加正样本权重
        >>> model.fit(X, y, sample_weight=sample_weight)

    筛选显著特征::

        >>> # 获取 p < 0.05 的显著特征
        >>> sig_features = model.get_significant_features(alpha=0.05)

    **注意事项**

    - 当 calculate_stats=True 时，会计算标准误差、z值、p值和VIF
    - 统计计算基于高斯假设，使用协方差矩阵估计
    - VIF 计算需要拟合截距项（fit_intercept=True）
    - VIF > 10 通常表示存在严重的多重共线性
    """

    def __init__(
        self,
        penalty: str = "l2",
        calculate_stats: bool = True,
        dual: bool = False,
        tol: float = 1e-4,
        C: float = 1.0,
        fit_intercept: bool = True,
        intercept_scaling: float = 1,
        class_weight: Optional[Union[dict, str]] = None,
        random_state: Optional[int] = None,
        solver: str = "lbfgs",
        max_iter: int = 100,
        multi_class: str = "auto",
        verbose: int = 0,
        warm_start: bool = False,
        n_jobs: Optional[int] = None,
        l1_ratio: Optional[float] = None,
        positive_woe_coef: Union[bool, str] = 'auto',
        target: Optional[str] = None,
    ):
        init_kwargs = {
            "penalty": penalty,
            "calculate_stats": calculate_stats,
            "dual": dual,
            "tol": tol,
            "C": C,
            "fit_intercept": fit_intercept,
            "intercept_scaling": intercept_scaling,
            "class_weight": class_weight,
            "random_state": random_state,
            "solver": solver,
            "max_iter": max_iter,
            "multi_class": multi_class,
            "verbose": verbose,
            "warm_start": warm_start,
            "n_jobs": n_jobs,
            "l1_ratio": l1_ratio,
        }
        init_kwargs = {k: v for k, v in init_kwargs.items() if k in _SKLEARN_LOGISTIC_PARAMS}

        super().__init__(**init_kwargs)
        self.calculate_stats = calculate_stats
        self.multi_class = multi_class
        self.positive_woe_coef = positive_woe_coef
        self.target = target

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Optional[Union[pd.Series, np.ndarray]] = None,
        sample_weight: Optional[np.ndarray] = None,
        **kwargs
    ) -> "LogisticRegression":
        """训练逻辑回归模型.

        支持两种调用方式:
        1. 常规方式: fit(X, y)
        2. scorecardpipeline风格: 在__init__中指定target，然后fit(X)

        在 sklearn LogisticRegression.fit() 的基础上，
        当 calculate_stats=True 时额外计算统计信息。

        **参数**

        :param X: 训练数据，形状 (n_samples, n_features)
            支持 numpy array 或 pandas DataFrame
            如果是DataFrame且y为None，会尝试从X中提取target列
        :param y: 目标变量，形状 (n_samples,)，可选
            二分类时为 0/1 或 -1/1
            如果为None且init中指定了target，则从X中提取
        :param sample_weight: 样本权重，形状 (n_samples,)
            默认 None，所有样本权重为1
        :param kwargs: 其他传递给父类 fit 方法的参数

        **返回**

        :return: self，训练好的模型实例

        **异常**

        :raises ValueError: 输入数据格式不正确
        :raises AssertionError: calculate_stats=False 时无法计算统计信息

        **参考样例**

        基本拟合::

            >>> model = LogisticRegression(calculate_stats=True)
            >>> model.fit(X_train, y_train)

        使用样本权重::

            >>> sample_weight = np.where(y_train == 1, 2.0, 1.0)
            >>> model.fit(X_train, y_train, sample_weight=sample_weight)

        scorecardpipeline风格::

            >>> model = LogisticRegression(target='label')
            >>> model.fit(X_train)  # 从X_train中提取'label'列作为y
        """
        # 处理 scorecardpipeline 风格：从 X 中提取 y
        if y is None and hasattr(self, 'target') and self.target is not None:
            if isinstance(X, pd.DataFrame) and self.target in X.columns:
                y = X[self.target]
                X = X.drop(columns=[self.target])

        # 保存特征名
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.tolist()
        else:
            self.feature_names_in_ = None

        # 如果不计算统计信息，直接调用父类方法
        apply_positive_woe_coef = self._should_apply_positive_woe_coef(X)

        if not self.calculate_stats:
            fitted_model = super().fit(X, y, sample_weight=sample_weight, **kwargs)
            if apply_positive_woe_coef:
                self.ensure_positive_woe_coefficients()
            return fitted_model

        # 转换稀疏矩阵
        X = self._convert_sparse_matrix(X)

        # 准备特征名列表（用于summary输出）
        if self.feature_names_in_ is not None:
            self.names_ = ["const"] + self.feature_names_in_
        else:
            self.names_ = ["const"] + [f"x{i}" for i in range(X.shape[1])]

        # 调用父类fit方法
        lr = super().fit(X, y, sample_weight=sample_weight, **kwargs)

        if apply_positive_woe_coef:
            self.ensure_positive_woe_coefficients()

        X_model = self._prepare_input_for_model(X)

        # 获取预测概率
        pred_probs = self._predict_proba_from_prepared_input(X_model)

        # 构建设计矩阵（添加截距列）
        if self.fit_intercept:
            X_design = np.hstack([np.ones((X_model.shape[0], 1)), X_model])
        else:
            X_design = X_model

        # 计算协方差矩阵和统计信息
        self._compute_statistics(X_design, pred_probs)

        return self

    def _should_apply_positive_woe_coef(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> bool:
        """判断是否需要对 WOE 模型做系数正向化."""
        if self.positive_woe_coef is True:
            return True
        if self.positive_woe_coef is False:
            return False
        if self.positive_woe_coef != 'auto':
            raise ValueError("positive_woe_coef 仅支持 True/False/'auto'")

        return isinstance(X, pd.DataFrame) and X.attrs.get('hscredit_encoding') == 'woe'

    def ensure_positive_woe_coefficients(
        self,
        X: Optional[Union[pd.DataFrame, np.ndarray]] = None,
    ) -> "LogisticRegression":
        """将 WOE 逻辑回归的负系数归一为正，并保持预测结果不变.

        做法是：对负系数对应的输入列乘以 -1，同时把系数改成绝对值。
        这样线性预测值保持不变，但模型摘要和评分卡解释更符合 WOE 场景。
        """
        check_is_fitted(self)

        if getattr(self, 'woe_coef_signs_', None) is None:
            coef_vector = np.asarray(self.coef_[0], dtype=float)
            self.raw_coef_ = self.coef_.copy()
            self.woe_coef_signs_ = np.where(coef_vector < 0, -1.0, 1.0)
            if np.any(self.woe_coef_signs_ < 0):
                self.coef_ = self.coef_.copy()
                self.coef_[0] = np.abs(coef_vector)
        elif X is None:
            return self

        if X is not None and self.calculate_stats:
            X_model = self._prepare_input_for_model(X)
            pred_probs = self._predict_proba_from_prepared_input(X_model)
            if self.fit_intercept:
                X_design = np.hstack([np.ones((X_model.shape[0], 1)), X_model])
            else:
                X_design = X_model
            self._compute_statistics(X_design, pred_probs)

        return self

    def _prepare_input_for_model(
        self,
        X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        """按 WOE 方向调整输入，使正向化后的系数仍保持原始预测结果."""
        X_model = self._convert_sparse_matrix(X)
        signs = getattr(self, 'woe_coef_signs_', None)
        if signs is None:
            return X_model

        if isinstance(X_model, pd.DataFrame):
            adjusted = X_model.copy()
            for feature_index, sign in enumerate(signs):
                if sign >= 0 or feature_index >= adjusted.shape[1]:
                    continue
                adjusted.iloc[:, feature_index] = adjusted.iloc[:, feature_index] * sign
            return adjusted

        adjusted = np.asarray(X_model).copy()
        adjusted = adjusted * signs
        return adjusted

    def _predict_proba_from_prepared_input(
        self,
        X_model: Union[pd.DataFrame, np.ndarray]
    ) -> np.ndarray:
        """对已经完成 WOE 方向处理的输入计算概率，避免重复变换."""
        decision = super().decision_function(X_model)
        if np.ndim(decision) == 1:
            positive_proba = scipy.special.expit(decision)
            return np.column_stack([1.0 - positive_proba, positive_proba])
        return scipy.special.softmax(decision, axis=1)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """预测概率，必要时自动应用 WOE 列方向调整."""
        X_model = self._prepare_input_for_model(X)
        return self._predict_proba_from_prepared_input(X_model)

    def predict_log_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """预测对数概率，必要时自动应用 WOE 列方向调整."""
        probabilities = self.predict_proba(X)
        return np.log(np.clip(probabilities, 1e-15, 1.0))

    def decision_function(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """计算决策函数，必要时自动应用 WOE 列方向调整."""
        X_model = self._prepare_input_for_model(X)
        return super().decision_function(X_model)

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """预测类别，必要时自动应用 WOE 列方向调整."""
        X_model = self._prepare_input_for_model(X)
        decision = super().decision_function(X_model)
        if np.ndim(decision) == 1:
            return np.where(decision > 0, self.classes_[1], self.classes_[0])
        return self.classes_[np.argmax(decision, axis=1)]

    def _compute_statistics(
        self,
        X_design: np.ndarray,
        pred_probs: np.ndarray
    ) -> None:
        """计算模型统计信息.

        计算协方差矩阵、标准误差、z统计量、p值和VIF。

        **参数**

        :param X_design: 设计矩阵（已添加截距列）
        :param pred_probs: 预测概率，形状 (n_samples, n_classes)
        """
        # 计算协方差矩阵: (X' * W * X)^(-1)
        p = np.prod(pred_probs, axis=1)
        
        # 使用伪逆矩阵处理奇异矩阵问题
        # 当存在多重共线性时，矩阵可能不可逆，使用 pinv 更稳健
        try:
            XTWX = (X_design * p[..., np.newaxis]).T @ X_design
            
            # 检查矩阵是否可逆
            cond_number = np.linalg.cond(XTWX)
            if cond_number > 1e10:  # 条件数过大，矩阵接近奇异
                import warnings
                warnings.warn(
                    f"协方差矩阵条件数过大 ({cond_number:.2e})，存在严重多重共线性。\n"
                    "建议使用 VIFSelector 或 CorrSelector 剔除高度相关的特征。\n"
                    "将使用伪逆矩阵计算统计量。",
                    UserWarning
                )
                self.cov_matrix_ = np.linalg.pinv(XTWX)
            else:
                self.cov_matrix_ = np.linalg.inv(XTWX)
                
        except np.linalg.LinAlgError as e:
            import warnings
            warnings.warn(
                f"协方差矩阵计算失败: {e}\n"
                "可能存在多重共线性问题，将使用伪逆矩阵。",
                UserWarning
            )
            # 使用伪逆矩阵作为备选
            XTWX = (X_design * p[..., np.newaxis]).T @ X_design
            self.cov_matrix_ = np.linalg.pinv(XTWX)

        # 计算标准误差
        std_err = np.sqrt(np.diag(self.cov_matrix_)).reshape(1, -1)

        # 分离截距和系数的标准误差
        if self.fit_intercept:
            self.std_err_intercept_ = std_err[:, 0]
            self.std_err_coef_ = std_err[:, 1:][0]

            # 计算z统计量
            self.z_intercept_ = self.intercept_ / self.std_err_intercept_
            self.z_coef_ = self.coef_ / self.std_err_coef_

            # 计算p值（基于高斯分布的双侧检验）
            self.p_val_intercept_ = scipy.stats.norm.sf(abs(self.z_intercept_)) * 2
            self.p_val_coef_ = scipy.stats.norm.sf(abs(self.z_coef_)) * 2
        else:
            # 没有截距时设为NaN
            self.std_err_intercept_ = np.array([np.nan])
            self.std_err_coef_ = std_err[0]

            self.z_intercept_ = np.array([np.nan])
            self.z_coef_ = self.coef_ / self.std_err_coef_

            self.p_val_intercept_ = np.array([np.nan])
            self.p_val_coef_ = scipy.stats.norm.sf(abs(self.z_coef_)) * 2

        # 计算VIF（方差膨胀因子）
        self.vif_ = self._compute_vif(X_design)
        
        # 检查是否存在多重共线性问题
        if hasattr(self, 'vif_') and self.vif_ is not None:
            # 检查是否有无限大的 VIF（完全共线性）
            inf_vif_count = np.sum(np.isinf(self.vif_))
            # 检查是否有高 VIF（> 10）
            high_vif_count = np.sum((self.vif_ > 10) & (self.vif_ != np.inf))
            
            if inf_vif_count > 0 or high_vif_count > 0:
                import warnings
                if inf_vif_count > 0:
                    warnings.warn(
                        f"检测到 {inf_vif_count} 个特征存在完全共线性（VIF=inf）。\n"
                        "这表明某些特征可以由其他特征完全线性表示。\n"
                        "建议使用 VIFSelector 或 CorrSelector 剔除这些特征。",
                        UserWarning
                    )
                elif high_vif_count > 0:
                    warnings.warn(
                        f"检测到 {high_vif_count} 个特征存在严重多重共线性（VIF > 10）。\n"
                        "这可能会影响模型系数的稳定性和解释性。\n"
                        "建议使用 VIFSelector 或 CorrSelector 降低特征相关性。",
                        UserWarning
                    )

    def _compute_vif(self, X_design: np.ndarray) -> np.ndarray:
        """计算方差膨胀因子 (VIF).

        VIF 用于检测多重共线性。VIF > 10 通常表示存在严重的多重共线性。

        **参数**

        :param X_design: 设计矩阵（已添加截距列）

        **返回**

        :return: vif，VIF值数组，长度与X_design列数相同
        """
        from statsmodels.stats.outliers_influence import variance_inflation_factor

        try:
            vif = [variance_inflation_factor(X_design, i) for i in range(X_design.shape[1])]
            return np.array(vif)
        except Exception:
            # 如果计算失败，返回NaN
            return np.full(X_design.shape[1], np.nan)

    @staticmethod
    def _convert_sparse_matrix(X: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """转换稀疏矩阵为密集矩阵.

        **参数**

        :param X: 输入数据，可能是稀疏矩阵

        **返回**

        :return: X，密集矩阵或原DataFrame
        """
        import scipy.sparse

        if scipy.sparse.issparse(X):
            return X.toarray()
        return X

    def get_feature_importances(self, importance_type: str = 'coef') -> pd.Series:
        """获取特征重要性.

        对于逻辑回归模型，使用系数绝对值作为特征重要性。

        :param importance_type: 重要性类型，默认'coef'
            - 'coef': 系数绝对值
            - 'p_value': 基于p值的重要性 (1 - p_value)
            - 'z_score': z统计量绝对值
        :return: 特征重要性Series
        """
        check_is_fitted(self, 'coef_')

        # 获取特征名称
        if self.feature_names_in_ is None:
            feature_names = [f'feature_{i}' for i in range(self.coef_.shape[1])]
        else:
            feature_names = self.feature_names_in_

        # 根据类型计算重要性
        if importance_type == 'coef':
            importances = np.abs(self.coef_[0])  # 取绝对值
        elif importance_type == 'p_value':
            if not hasattr(self, 'p_val_coef_'):
                raise ValueError("未计算p值，请在fit时设置calculate_stats=True")
            importances = 1 - self.p_val_coef_[0]  # p值越小越重要
        elif importance_type == 'z_score':
            if not hasattr(self, 'z_coef_'):
                raise ValueError("未计算z统计量，请在fit时设置calculate_stats=True")
            importances = np.abs(self.z_coef_[0])
        else:
            raise ValueError(f"不支持的重要性类型: {importance_type}")

        # 创建Series
        importance_series = pd.Series(
            importances,
            index=feature_names,
            name='importance'
        ).sort_values(ascending=False)

        self._feature_importances = importance_series

        return importance_series

    @property
    def feature_importances_(self) -> np.ndarray:
        """特征重要性属性 (兼容sklearn风格).

        直接从内部模型获取系数绝对值，避免缓存逻辑在clone后出错。
        """
        check_is_fitted(self, 'coef_')
        return np.abs(self.coef_[0])

    def summary(self) -> pd.DataFrame:
        """获取回归结果的统计摘要.

        返回包含系数、标准误差、z统计量、p值、置信区间和VIF的DataFrame。

        **返回**

        :return: summary_df，统计摘要表，包含以下列：
            - Coef.: 回归系数
            - Std.Err: 标准误差
            - z: z统计量（系数/标准误差）
            - P>|z|: p值
            - [0.025: 95%置信区间下限
            - 0.975]: 95%置信区间上限
            - VIF: 方差膨胀因子

        **异常**

        :raises AssertionError: 如果训练时 calculate_stats=False

        **参考样例**

        基本使用::

            >>> model = LogisticRegression(calculate_stats=True)
            >>> model.fit(X, y)
            >>> summary = model.summary()
            >>> print(summary)

        筛选显著变量::

            >>> # 筛选 p < 0.05 的显著变量
            >>> significant = summary[summary['P>|z|'] < 0.05]
            >>> print(significant[['Coef.', 'P>|z|', 'VIF']])
        """
        check_is_fitted(self)

        if not hasattr(self, "std_err_coef_"):
            msg = "统计信息未计算。解决方法:\n"
            msg += "  1. 使用 model.fit(X, y) 重新训练（calculate_stats=True）\n"
            msg += "  2. 初始化时设置 calculate_stats=True"
            raise AssertionError(msg)

        # 构建统计表
        data = {
            "Coef.": np.concatenate([self.intercept_.flatten(), self.coef_.flatten()]),
            "Std.Err": np.concatenate([self.std_err_intercept_.flatten(), self.std_err_coef_.flatten()]),
            "z": np.concatenate([self.z_intercept_.flatten(), self.z_coef_.flatten()]),
            "P>|z|": np.concatenate([self.p_val_intercept_.flatten(), self.p_val_coef_.flatten()]),
        }

        summary_df = pd.DataFrame(data, index=self.names_)

        # 计算95%置信区间
        summary_df["[0.025"] = summary_df["Coef."] - 1.96 * summary_df["Std.Err"]
        summary_df["0.975]"] = summary_df["Coef."] + 1.96 * summary_df["Std.Err"]

        # 添加VIF
        summary_df["VIF"] = self.vif_

        return summary_df

    def summary_with_desc(self, feature_map: Optional[dict] = None) -> pd.DataFrame:
        """获取带特征描述的统计摘要.

        在 summary() 基础上增加特征描述列。

        **参数**

        :param feature_map: 特征描述字典，格式为 {特征名: 描述}，默认 None

        **返回**

        :return: summary_df，带描述的统计摘要表，增加 Features 和 Describe 列

        **参考样例**

        使用特征描述::

            >>> feature_map = {
            ...     'age': '年龄',
            ...     'income': '收入',
            ... }
            >>> summary = model.summary_with_desc(feature_map)
            >>> print(summary[['Features', 'Describe', 'Coef.', 'P>|z|']])
        """
        summary_df = self.summary().reset_index().rename(columns={"index": "Features"})

        if feature_map is not None and len(feature_map) > 0:
            summary_df.insert(
                loc=1,
                column="Describe",
                value=[feature_map.get(c, "") for c in summary_df["Features"]]
            )

        return summary_df

    def get_significant_features(
        self,
        alpha: float = 0.05,
        include_intercept: bool = False
    ) -> pd.DataFrame:
        """获取统计显著的特征.

        根据p值筛选显著特征。

        **参数**

        :param alpha: 显著性水平，默认 0.05
            p < alpha 的特征被认为是显著的
        :param include_intercept: 是否包含截距项，默认 False

        **返回**

        :return: significant_df，显著特征的统计信息

        **参考样例**

        筛选显著特征::

            >>> # 获取 p < 0.01 的显著特征
            >>> sig_features = model.get_significant_features(alpha=0.01)
            >>> print(sig_features)
        """
        summary = self.summary()

        # 筛选显著特征
        mask = summary["P>|z|"] < alpha

        # 是否包含截距
        if not include_intercept:
            mask = mask & (summary.index != "const")

        return summary[mask]

    def check_multicollinearity(self, threshold: float = 10.0) -> pd.DataFrame:
        """检查多重共线性.

        基于VIF值检测多重共线性。VIF > threshold 表示存在共线性问题。

        **参数**

        :param threshold: VIF阈值，默认 10.0
            VIF > threshold 被认为存在多重共线性

        **返回**

        :return: high_vif_df，VIF值超过阈值的特征信息

        **参考样例**

        检查共线性::

            >>> # 检查 VIF > 5 的特征
            >>> collinear = model.check_multicollinearity(threshold=5.0)
            >>> print(collinear)
        """
        summary = self.summary()

        # 排除截距
        mask = (summary.index != "const") & (summary["VIF"] > threshold)

        return summary[mask][["Coef.", "VIF"]].sort_values("VIF", ascending=False)

    def __getstate__(self):
        """支持 pickle 序列化.

        确保 dill/joblib 等序列化引擎能正确处理继承自 sklearn 的类。
        """
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        """支持 pickle 反序列化."""
        self.__dict__.update(state)
        # 确保父类状态正确恢复
        if not hasattr(self, 'classes_'):
            # 如果模型未拟合，不需要额外处理
            pass
