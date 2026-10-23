"""方差筛选器.

移除低方差特征。

**参考样例**

>>> from hscredit.core.selectors import VarianceSelector
>>> import pandas as pd
>>> X = pd.DataFrame({'a': [1,2,3], 'b': [1,1,1], 'c': [1,2,3]})  # b为常量特征，方差为0
>>> selector = VarianceSelector(threshold=0.1)  # 移除方差<0.1的特征
>>> selector.fit(X)
>>> print(selector.selected_features_)
['a', 'c']
"""

from typing import Union, List, Optional, Dict, Any
import numpy as np
import pandas as pd
from .base import BaseFeatureSelector
from ._statistical_utils import record_conditions, record_counts, validate_real


def _compute_variance_feature(task):
    """计算单列总体方差与峰值差。"""
    feature, series = task
    return feature, series.var(ddof=0), series.max() - series.min()


class VarianceSelector(BaseFeatureSelector):
    """方差筛选器.

    移除方差小于等于阈值的特征。
    常用于移除常量特征或近似常量特征。

    **参数**

    :param target: 目标变量列名，默认为 'target'。在 scorecardpipeline 风格中用于
        从 DataFrame 中提取目标列；在 sklearn 风格中若 fit 传入了 y 参数则优先使用 y

    :type target: str
    :param include: 默认值为 ``None`` 。强制保留的特征列表，这些特征无论如何都会被保留
    :type include: Optional[List[str]]
    :param exclude: 默认值为 ``None`` 。强制剔除的特征列表，这些特征无论如何都会被剔除
    :type exclude: Optional[List[str]]
    :param force_drop: 默认值为 ``None`` 。强制剔除的特征列表，效果与 exclude 合并
    :type force_drop: Optional[List[str]]
    :param n_jobs: 并行总预算，默认 ``-1`` 自动按物理核与工作量调度； ``1`` 或 ``None`` 为串行，
        正整数为 worker 上限， ``0 < n_jobs < 1`` 为物理核比例（向上取整）。
        不接受布尔值、 ``0`` 或小于 ``-1`` 的值；嵌套任务共享总预算。

    :type n_jobs: Optional[Union[int, float]]
    :param binner: 默认值为 ``None`` 。可选的已配置分箱器实例，支持已训练或待训练状态，不接受分箱器类
    :type binner: Optional[Any]
    :param bin_params: 默认值为 ``None`` 。可选的 ``OptimalBinning`` 构造参数字典。未传入 ``binner`` 时
        基类使用该字典创建分箱器；同时传入时 ``binner`` 优先

    :type bin_params: Optional[Dict[str, Any]]
    :param parallel_backend: joblib 后端名称，默认 ``None`` 由模块选择；常用 ``"threading"`` 、
        ``"loky"`` 、 ``"multiprocessing"`` ，也可使用已注册后端。
        显式后端必须与 ``parallel_config`` 中的共享内存要求兼容。

    :type parallel_backend: Optional[str]
    :param parallel_config: 默认值为 ``None`` 。并行配置字典或 ``None`` 。支持 ``adaptive`` 、 ``prefer`` 、 ``require`` 、
        ``batch_size`` 、 ``pre_dispatch`` 、 ``max_nbytes`` 、 ``mmap_mode`` 、
        ``temp_folder`` 、 ``timeout`` 、 ``verbose`` 、 ``inner_max_num_threads`` 、
        ``backend_kwargs`` ；并行数和后端应分别通过对应公共参数传入。

    :type parallel_config: Optional[Dict[str, Any]]
    :param target_rm: 默认False保留输入中的目标列；仅显式True时移除，不影响拟合时目标隔离
    :type target_rm: bool

    :param threshold: 方差阈值，默认为0.0

        - 0.0: 移除常量特征（方差为0）

        - 其他值: 移除方差小于等于该值的特征

    :type threshold: float

    **参考样例**

    ::

        >>> from hscredit.core.selectors import VarianceSelector
        >>> import pandas as pd
        >>> X = pd.DataFrame({'a': [1,2,3], 'b': [1,1,1], 'c': [1,2,3]})
        >>> selector = VarianceSelector(threshold=0.1)
        >>> selector.fit(X)
        >>> print(selector.selected_features_)
        ['a', 'c']

    **注意**

    方差筛选为无监督方法，不使用标签 ``y`` ；方差受量纲影响，不同尺度特征建议先标准化
    再比较，否则大量纲特征会因方差天然偏大而被保留。

    **引用**

    对齐 sklearn ``VarianceThreshold`` ：
    https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.VarianceThreshold.html
    """

    method_name = "方差筛选"

    def __init__(
        self,
        threshold: float = 0.0,
        target: str = "target",
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        force_drop: Optional[List[str]] = None,
        n_jobs: Optional[Union[int, float]] = -1,
        binner: Optional[Any] = None,
        bin_params: Optional[Dict[str, Any]] = None,
        parallel_backend: Optional[str] = None,
        parallel_config: Optional[Dict[str, Any]] = None,
        target_rm: bool = False,
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
            bin_params=bin_params,
            parallel_backend=parallel_backend,
            parallel_config=parallel_config,
            target_rm=target_rm,
        )

    def _check_input(self, X, y=None):
        validate_real(self.threshold, "方差阈值", minimum=0)
        return super()._check_input(X, y)

    def _fit_impl(
        self,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, np.ndarray]],
    ) -> None:
        """拟合方差筛选器。

        :param X: 输入特征DataFrame
        :param y: 目标变量（此筛选器不需要）
        """
        self._get_feature_names(X)

        self._validate_parallel_configuration()
        record_counts(self, X)
        self.threshold_ = self.threshold
        self.score_name_, self.score_direction_ = "方差", "越大越好"
        non_numeric = [column for column in X if not pd.api.types.is_numeric_dtype(X[column])]
        if non_numeric:
            raise ValueError(f"方差筛选只支持数值字段，请先编码: {non_numeric}")
        self.variances_ = X.var(axis=0, ddof=0, numeric_only=False).reindex(X.columns).astype(float)
        self.scores_ = self.variances_.copy()
        peak_to_peak = (X.max(axis=0) - X.min(axis=0)).reindex(X.columns)
        self.ranges_ = peak_to_peak

        # 根据阈值筛选
        if self.threshold == 0:
            # 极差仅辅助识别数值舍入形成的伪常量，不覆盖真实方差指标。
            selected_mask = (self.variances_.fillna(0) > 0) & (peak_to_peak.fillna(0) > 0)
        else:
            selected_mask = self.variances_ > self.threshold

        # 选择方差大于阈值的特征
        record_conditions(self, X.columns, 方差达标=selected_mask)
        self.selected_features_ = X.columns[selected_mask].tolist()

        # 构建详细的dropped_记录，包含方差值
        dropped_cols = X.columns[~selected_mask].tolist()
        if len(dropped_cols) > 0:
            self.dropped_ = pd.DataFrame(
                {
                    "特征": dropped_cols,
                    "剔除原因": [f"方差({self.scores_[col]:.6f}) <= 阈值({self.threshold})" for col in dropped_cols],
                    "方差": [self.scores_[col] for col in dropped_cols],
                    "阈值": [self.threshold] * len(dropped_cols),
                }
            )
        else:
            self.dropped_ = pd.DataFrame(columns=["特征", "剔除原因", "方差", "阈值"])
