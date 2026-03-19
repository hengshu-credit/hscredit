"""IV值筛选器.

使用信息价值（IV）进行特征筛选，是金融风控场景的核心筛选方法。
"""

from typing import Union, List, Optional
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .base import BaseFeatureSelector


def _compute_iv_single(x: np.ndarray, y: np.ndarray, regularization: float = 1.0) -> float:
    """计算单个特征的IV值。

    :param x: 特征值数组
    :param y: 目标变量数组
    :param regularization: 正则化参数，避免除零
    :return: IV值
    """
    # 处理缺失值 - 兼容category和object类型
    # 先转换为object类型，然后用isnull判断
    if isinstance(x, pd.Series):
        has_missing = x.isnull().values
    else:
        # 如果是numpy数组，尝试转换为Series以使用isnull
        try:
            has_missing = pd.Series(x).isnull().values
        except:
            # 如果转换失败，使用pd.isnull直接判断
            has_missing = pd.isnull(x)
    
    valid = ~has_missing
    x_valid = x[valid]
    y_valid = y[valid]
    
    if len(x_valid) == 0:
        return 0.0
    
    # 获取唯一值
    uniques = np.unique(x_valid)
    n_cats = len(uniques)
    
    if n_cats <= 1:
        return 0.0
    
    # 统计好坏样本
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
    
    # 避免极端值
    bad_pos = (event_rates + nonevent_rates) == (2 * regularization + 1)
    event_rates /= event_tot
    nonevent_rates /= nonevent_tot
    
    # 计算IV
    ivs = (event_rates - nonevent_rates) * np.log(
        np.maximum(event_rates, 1e-10) / np.maximum(nonevent_rates, 1e-10)
    )
    ivs[bad_pos] = 0.0
    
    return np.sum(ivs).item()


class IVSelector(BaseFeatureSelector):
    """IV值筛选器.

    使用信息价值（Information Value）筛选特征。
    IV是金融风控中衡量特征预测能力的核心指标。

    IV值解释:
    - < 0.02: 无预测能力
    - 0.02 - 0.1: 弱预测能力
    - 0.1 - 0.3: 中等预测能力
    - 0.3 - 0.5: 强预测能力
    - > 0.5: 极强预测能力（可能过拟合）

    **支持的数据类型:**
    - 数值型特征（int, float）
    - 类别型特征（object, category）

    **参数**

    :param threshold: IV阈值，默认为0.02
        - 0.02: 仅保留IV值大于0.02的特征
    :param target: 目标变量列名，默认为'target'
    :param regularization: 正则化参数，默认为1.0
    :param n_jobs: 并行计算的任务数

    **示例**

    ::

        >>> from hscredit.core.selection import IVSelector
        >>> import pandas as pd
        >>> X = pd.DataFrame({
        ...     'income': [5000, 8000, 12000, 2000, 15000],
        ...     'age': [25, 35, 45, 55, 23],
        ... })
        >>> y = pd.Series([0, 0, 1, 0, 1])
        >>> selector = IVSelector(threshold=0.02)
        >>> selector.fit(X, y)
        >>> print(selector.select_columns_)
        >>> print(selector.scores_)  # 查看IV值
    """

    def __init__(
        self,
        threshold: float = 0.02,
        target: str = 'target',
        regularization: float = 1.0,
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        n_jobs: int = 1,
    ):
        super().__init__(target=target, threshold=threshold, include=include, exclude=exclude, n_jobs=n_jobs)
        self.regularization = regularization
        self.method_name = 'IV值筛选'

    def _fit_impl(
        self,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, np.ndarray]],
    ) -> None:
        """拟合IV值筛选器。

        :param X: 输入特征DataFrame
        :param y: 目标变量
        """
        if y is None:
            if self.target not in X.columns:
                raise ValueError(f"需要传入y或X中包含{self.target}列")
            y = X[self.target].values
            X = X.drop(columns=self.target)

        self._get_feature_names(X)

        # 编码类别变量 - 支持 object 和 category 类型
        X_encoded = X.copy()
        for col in X.columns:
            # 检查是否为类别型变量（object 或 category 类型）
            if X[col].dtype.name in ['object', 'category']:
                X_encoded[col] = pd.factorize(X[col])[0]

        y = np.asarray(y)

        # 计算IV值
        if self.n_jobs == 1:
            iv_values = np.array([
                _compute_iv_single(X_encoded[col].values, y, self.regularization)
                for col in X_encoded.columns
            ])
        else:
            iv_values = np.array(
                Parallel(n_jobs=self.n_jobs)(
                    delayed(_compute_iv_single)(X_encoded[col].values, y, self.regularization)
                    for col in X_encoded.columns
                )
            )

        self.scores_ = pd.Series(iv_values, index=X.columns)

        # 选择IV值大于等于阈值的特征
        selected_mask = iv_values >= self.threshold
        self.selected_features_ = X.columns[selected_mask].tolist()
        self._drop_reason = f'IV值 <= {self.threshold}'

    def get_iv_interpretation(self) -> pd.DataFrame:
        """获取IV值的中文解释。

        :return: 包含IV值及解释的DataFrame
        """
        if not hasattr(self, 'scores_'):
            return pd.DataFrame()

        def interpret_iv(iv):
            if iv < 0.02:
                return '无预测能力'
            elif iv < 0.1:
                return '弱预测能力'
            elif iv < 0.3:
                return '中等预测能力'
            elif iv < 0.5:
                return '强预测能力'
            else:
                return '极强预测能力（可能过拟合）'

        df = pd.DataFrame({
            '特征': self.scores_.index,
            'IV值': self.scores_.values,
            '预测能力': [interpret_iv(iv) for iv in self.scores_.values]
        })
        return df.sort_values('IV值', ascending=False)
