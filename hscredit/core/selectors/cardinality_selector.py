"""基数筛选器.

移除基数（唯一值数量）过高的类别型特征。

**参考样例**

>>> from hscredit.core.selectors import CardinalitySelector
>>> import pandas as pd
>>> X = pd.DataFrame({
...     'city': ['北京', '上海', '广州', '北京', '深圳'],
...     'id': [1, 2, 3, 4, 5],  # 高基数
... })
>>> selector = CardinalitySelector(threshold=4)
>>> selector.fit(X)
"""

from typing import Union, List, Optional, Dict, Any
import numpy as np
import pandas as pd

from .base import BaseFeatureSelector
from ._statistical_utils import record_conditions, record_counts, validate_real


def _compute_cardinality_feature(task):
    """计算单列唯一值数量。"""
    feature, series, dropna = task
    return feature, series.nunique(dropna=dropna)


class CardinalitySelector(BaseFeatureSelector):
    """基数筛选器.

    移除基数高于阈值的类别型特征。
    高基数特征可能导致过拟合和计算问题。

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

    :param threshold: 基数阈值，默认为10

        - 10: 移除唯一值数量超过10的类别型特征

    :type threshold: int
    :param dropna: 是否在统计唯一值数量时排除NaN，默认为True
    :type dropna: bool

    **参考样例**

    ::

        >>> from hscredit.core.selectors import CardinalitySelector
        >>> import pandas as pd
        >>> X = pd.DataFrame({
        ...     'city': ['北京', '上海', '广州', '北京', '深圳'],
        ...     'id': [1, 2, 3, 4, 5],  # 高基数
        ... })
        >>> selector = CardinalitySelector(threshold=4)
        >>> selector.fit(X)
    """

    method_name = "基数筛选"

    def __init__(
        self,
        threshold: int = 10,
        dropna: bool = True,
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
        self.dropna = dropna

    def _check_input(self, X, y=None):
        validate_real(self.threshold, "基数阈值", minimum=0)
        if int(self.threshold) != self.threshold:
            raise ValueError("基数阈值必须是非负整数")
        if not isinstance(self.dropna, (bool, np.bool_)):
            raise ValueError("dropna 必须是布尔值")
        return super()._check_input(X, y)

    def _fit_impl(
        self,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, np.ndarray]],
    ) -> None:
        """拟合基数筛选器。

        :param X: 输入特征DataFrame
        :param y: 目标变量（此筛选器不需要）
        """
        self._get_feature_names(X)

        self._validate_parallel_configuration()
        record_counts(self, X)
        self.threshold_ = self.threshold
        self.score_name_, self.score_direction_ = "唯一值数量", "越小越好"
        cardinalities = X.nunique(axis=0, dropna=self.dropna).reindex(X.columns)
        self.scores_ = cardinalities
        self.cardinalities_ = cardinalities.copy()

        # 选择基数低于阈值的特征
        selected_mask = cardinalities <= self.threshold
        record_conditions(self, X.columns, 基数达标=selected_mask)
        self.selected_features_ = X.columns[selected_mask].tolist()

        # 构建详细的dropped_记录，包含基数信息
        dropped_cols = X.columns[~selected_mask].tolist()
        if len(dropped_cols) > 0:
            self.dropped_ = pd.DataFrame(
                {
                    "特征": dropped_cols,
                    "剔除原因": [f"唯一值数量({self.scores_[col]}) > 阈值({self.threshold})" for col in dropped_cols],
                    "唯一值数量": [self.scores_[col] for col in dropped_cols],
                    "阈值": [self.threshold] * len(dropped_cols),
                }
            )
        else:
            self.dropped_ = pd.DataFrame(columns=["特征", "剔除原因", "唯一值数量", "阈值"])
