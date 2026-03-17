"""VIF筛选器.

使用方差膨胀因子（VIF）检测和移除多重共线性特征。
"""

from typing import Union, List, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from joblib import Parallel, delayed

from .base import BaseFeatureSelector


def _compute_vif_single(x: np.ndarray, idx: int) -> float:
    """计算单个特征的VIF值。

    :param x: 特征矩阵
    :param idx: 特征索引
    :return: VIF值
    """
    n_features = x.shape[1]
    if n_features <= 1:
        return 1.0  # 只有一个特征时，VIF=1
    
    # 获取其他特征
    mask = np.ones(n_features, dtype=bool)
    mask[idx] = False
    x_other = x[:, mask]
    x_target = x[:, idx]
    
    # 处理缺失值
    valid = ~(np.isnan(x_target) | np.any(np.isnan(x_other), axis=1))
    if valid.sum() < 2:
        return np.inf
    
    x_other = x_other[valid]
    x_target = x_target[valid]
    
    # 检查目标特征是否为常数
    if np.std(x_target) < 1e-10:
        return 1.0  # 常数特征VIF=1
    
    # 检查其他特征是否全为常数
    if np.all(np.std(x_other, axis=0) < 1e-10):
        return 1.0  # 其他特征都是常数，无法预测目标特征
    
    # 线性回归
    try:
        lr = LinearRegression(fit_intercept=False)
        lr.fit(x_other, x_target)
        y_pred = lr.predict(x_other)
        
        # 计算VIF
        ss_res = np.sum((x_target - y_pred) ** 2)
        ss_tot = np.sum((x_target - np.mean(x_target)) ** 2)
        
        if ss_tot < 1e-10:
            return 1.0
        
        r2 = 1 - ss_res / ss_tot
        
        # 处理r2接近1或大于1的情况
        if r2 >= 1.0:
            return np.inf
        elif r2 < 0:
            # r2 < 0 表示模型比均值还差，VIF很小
            return 1.0
        else:
            vif = 1 / (1 - r2)
            return vif
    except Exception:
        return np.inf


class VIFSelector(BaseFeatureSelector):
    """VIF筛选器.

    使用方差膨胀因子（VIF）检测多重共线性。
    VIF值越高，表示特征与其他特征的多重共线性越严重。
    在金融风控中，通常认为VIF > 4存在多重共线性问题。

    **算法说明：**

    采用迭代剔除策略：
    1. 计算所有特征的VIF值
    2. 如果最大VIF > threshold，剔除VIF最大的特征
    3. 重新计算剩余特征的VIF
    4. 重复步骤2-3，直到所有VIF <= threshold

    这种方法避免了"一刀切"地剔除所有高VIF特征的问题。

    **参数**

    :param threshold: VIF阈值，默认为4.0
        - 4.0: 移除VIF值超过4的特征
        - 范围: 正数
    :param missing: 缺失值填充值，默认为-1
    :param max_iter: 最大迭代次数，默认为100
    :param n_jobs: 并行计算的任务数
    :param verbose: 是否显示详细过程，默认为False

    **示例**

    ::

        >>> from hscredit.core.selection import VIFSelector
        >>> import pandas as pd
        >>> import numpy as np
        >>> X = pd.DataFrame({
        ...     'a': [1, 2, 3, 4, 5],
        ...     'b': [1, 2, 3, 4, 5],  # 与a完全相关
        ...     'c': [5, 4, 3, 2, 1]
        ... })
        >>> selector = VIFSelector(threshold=4.0)
        >>> selector.fit(X)
        >>> print(selector.selected_features_)
        ['a', 'c']  # 或 ['b', 'c']，保留其中一个高相关特征
    """

    def __init__(
        self,
        threshold: float = 4.0,
        missing: float = -1.0,
        max_iter: int = 100,
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        n_jobs: int = 1,
        verbose: bool = False,
    ):
        super().__init__(threshold=threshold, include=include, exclude=exclude, n_jobs=n_jobs)
        self.missing = missing
        self.max_iter = max_iter
        self.verbose = verbose
        self.method_name = 'VIF筛选'

    def _compute_vif_all(self, X: pd.DataFrame) -> pd.Series:
        """计算所有特征的VIF值。

        :param X: 特征DataFrame
        :return: VIF值Series
        """
        x_filled = X.fillna(self.missing).values
        n_features = x_filled.shape[1]
        
        if n_features == 0:
            return pd.Series(dtype=float)
        
        if self.n_jobs == 1:
            vif_values = np.array([
                _compute_vif_single(x_filled, i) 
                for i in range(n_features)
            ])
        else:
            vif_values = np.array(
                Parallel(n_jobs=self.n_jobs)(
                    delayed(_compute_vif_single)(x_filled, i)
                    for i in range(n_features)
                )
            )
        
        return pd.Series(vif_values, index=X.columns)

    def _fit_impl(
        self,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, np.ndarray]],
    ) -> None:
        """拟合VIF筛选器。

        采用迭代剔除策略，每次只剔除VIF最大的特征。

        :param X: 输入特征DataFrame
        :param y: 目标变量（此筛选器不需要）
        """
        self._get_feature_names(X)
        
        # 保留的特征列表
        remaining_features = X.columns.tolist()
        # 被剔除的特征及原因
        dropped_features = []
        dropped_reasons = []
        # 记录每次迭代的VIF值
        vif_history = []
        
        # 迭代剔除
        for iteration in range(self.max_iter):
            if len(remaining_features) == 0:
                break
            
            # 计算当前所有特征的VIF
            X_current = X[remaining_features]
            vif_series = self._compute_vif_all(X_current)
            vif_history.append(vif_series.copy())
            
            # 找到VIF最大的特征
            max_vif = vif_series.max()
            max_feature = vif_series.idxmax()
            
            if self.verbose:
                print(f"迭代 {iteration + 1}: 最大VIF = {max_vif:.4f} (特征: {max_feature})")
            
            # 如果最大VIF <= threshold，停止
            if max_vif <= self.threshold:
                if self.verbose:
                    print(f"所有特征VIF <= {self.threshold}，停止迭代")
                break
            
            # 剔除VIF最大的特征
            remaining_features.remove(max_feature)
            dropped_features.append(max_feature)
            dropped_reasons.append(f'VIF={max_vif:.4f} (第{iteration + 1}轮剔除)')
            
            if self.verbose:
                print(f"  剔除特征: {max_feature}")
        
        # 保存结果
        self.selected_features_ = remaining_features
        self.removed_features_ = dropped_features
        
        # 保存最终的VIF值作为scores_
        if len(remaining_features) > 0:
            final_vif = self._compute_vif_all(X[remaining_features])
            self.scores_ = final_vif
        else:
            self.scores_ = pd.Series(dtype=float)
        
        # 记录剔除历史
        self.vif_history_ = vif_history
        self.n_iterations_ = iteration + 1
        
        # 构建dropped_ DataFrame
        if len(dropped_features) > 0:
            self.dropped_ = pd.DataFrame({
                '特征': dropped_features,
                '剔除原因': dropped_reasons
            })
        else:
            self.dropped_ = pd.DataFrame(columns=['特征', '剔除原因'])
        
        self._drop_reason = f'VIF值 > {self.threshold}'
        
        if self.verbose:
            print(f"\nVIF筛选完成:")
            print(f"  迭代次数: {self.n_iterations_}")
            print(f"  保留特征: {len(self.selected_features_)}")
            print(f"  剔除特征: {len(self.removed_features_)}")
            if len(self.selected_features_) > 0:
                print(f"  最终最大VIF: {self.scores_.max():.4f}")
