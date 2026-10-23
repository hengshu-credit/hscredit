"""单一值筛选器.

移除单一值（众数）占比过高的特征。

**参考样例**

>>> from hscredit.core.selectors import ModeSelector
>>> import pandas as pd
>>> X = pd.DataFrame({
...     'a': [1, 1, 1, 1, 2],    # 众数(1)占比80%
...     'b': [1, 2, 3, 4, 5],    # 众数(1)占比20%
...     'c': [1, 1, 1, 1, 1]     # 常量特征，众数占比100%
... })
>>> selector = ModeSelector(threshold=0.8)  # 移除众数占比>=80%的特征
>>> selector.fit(X)
>>> print(selector.selected_features_)
['b']
"""

from typing import Union, List, Optional, Dict, Any
import numpy as np
import pandas as pd

from .base import BaseFeatureSelector
from ._statistical_utils import record_conditions, record_counts, validate_real


def _compute_mode_ratio(series: pd.Series, dropna: bool = True) -> float:
    """计算众数占比。

    :param series: 输入序列
    :param dropna: 是否排除缺失值
    :return: 众数占比
    """
    if len(series) == 0:
        return 1.0

    summary = series.value_counts(dropna=dropna)
    if len(summary) == 0:
        return 1.0

    denominator = int(series.notna().sum()) if dropna else len(series)
    if denominator == 0:
        return 1.0
    return summary.iloc[0] / denominator


def _compute_mode_feature(task):
    """计算单列众数占比并携带特征名返回。"""
    feature, series, dropna = task
    return feature, _compute_mode_ratio(series, dropna)


class ModeSelector(BaseFeatureSelector):
    """单一值筛选器.

    移除众数占比大于等于阈值的特征。
    用于过滤掉取值过于集中、区分度低的特征。

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

    :param threshold: 单一值占比阈值，默认为0.95

        - 0.95: 移除单一值占比达到或超过95%的特征

        - 范围: 0-1之间的浮点数

    :type threshold: float
    :param dropna: 是否在计算众数占比时排除NaN，默认为True
    :type dropna: bool
    :param n_jobs: 并行总预算，默认 ``-1`` 自动按物理核与工作量调度； ``1`` 或 ``None`` 为串行，
        正整数为 worker 上限， ``0 < n_jobs < 1`` 为物理核比例（向上取整）。
        不接受布尔值、 ``0`` 或小于 ``-1`` 的值；嵌套任务共享总预算。

    :type n_jobs: Optional[Union[int, float]]

    **参考样例**

    ::

        >>> from hscredit.core.selectors import ModeSelector
        >>> import pandas as pd
        >>> X = pd.DataFrame({
        ...     'a': [1, 1, 1, 1, 2],
        ...     'b': [1, 2, 3, 4, 5],
        ...     'c': [1, 1, 1, 1, 1]
        ... })
        >>> selector = ModeSelector(threshold=0.8)
        >>> selector.fit(X)
        >>> print(selector.selected_features_)
        ['b']
    """

    method_name = "单一值筛选"

    def __init__(
        self,
        threshold: float = 0.95,
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
        validate_real(self.threshold, "众数占比阈值", minimum=0, maximum=1)
        if not isinstance(self.dropna, (bool, np.bool_)):
            raise ValueError("dropna 必须是布尔值")
        return super()._check_input(X, y)

    def _fit_impl(
        self,
        X: pd.DataFrame,
        y: Optional[Union[pd.Series, np.ndarray]],
    ) -> None:
        """拟合单一值筛选器。

        :param X: 输入特征DataFrame
        :param y: 目标变量（此筛选器不需要）
        """
        self._get_feature_names(X)

        self._validate_parallel_configuration()
        record_counts(self, X)
        self.threshold_ = self.threshold
        self.score_name_, self.score_direction_ = "众数占比", "越小越好"
        mode_values, mode_counts, mode_ratios = [], [], []
        for column in X.columns:
            counts = X[column].value_counts(dropna=self.dropna)
            count = int(counts.iloc[0]) if len(counts) else 0
            denominator = int(self.valid_counts_[column]) if self.dropna else len(X)
            mode_values.append(counts.index[0] if len(counts) else np.nan)
            mode_counts.append(count)
            mode_ratios.append(count / denominator if denominator else 1.0)
        self.mode_values_ = pd.Series(mode_values, index=X.columns, dtype=object)
        self.mode_counts_ = pd.Series(mode_counts, index=X.columns, dtype=np.int64)
        self.denominator_counts_ = self.valid_counts_.copy() if self.dropna else self.total_counts_.copy()
        mode_ratios = pd.Series(mode_ratios, index=X.columns, dtype=float)

        self.scores_ = mode_ratios

        # 选择众数占比低于阈值的特征
        selected_mask = mode_ratios < self.threshold
        record_conditions(self, X.columns, 众数占比达标=selected_mask)
        self.selected_features_ = X.columns[selected_mask].tolist()
        self._drop_reason = f"单一值占比 >= {self.threshold:.2%}"
